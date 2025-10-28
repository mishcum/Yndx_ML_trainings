import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(4, 2))
        self.layers.load_state_dict(OrderedDict([('0.weight',
                                    torch.tensor([[ 0.2386,  0.5018,  3.0809,  1.3327],
                                            [ 0.1455, -0.1592,  0.3834, -0.2027],
                                            [-0.0480,  0.0677, -0.4613,  0.1564],
                                            [-0.1892, -0.0508, -2.6197, -1.3168]], dtype=torch.float32)),
                                    ('0.bias', torch.tensor([ 0.1384, -0.4760, -0.1760,  0.9781], dtype=torch.float32)),
                                    ('3.weight',
                                    torch.tensor([[-1.7106,  0.0190,  0.1120,  1.3720],
                                            [ 1.7207,  0.0536, -0.0219, -1.6665]], dtype=torch.float32)),
                                    ('3.bias', torch.tensor([ 0.1195, -0.3039], dtype=torch.float32))]))
    def forward(self, X):
        return self.layers(X)
    
model = Model().eval()

def to_one_hot(y_tensor, ndims):
    """ helper: take an integer vector and convert it to 1-hot matrix. """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def predict_probs(states):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :returns: numpy array of shape [batch, n_actions]
    """
    model.eval()
    with torch.no_grad():
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        logits = model(states)

    probs = torch.softmax(logits, dim=1).cpu().numpy()
    assert probs is not None, "probs is not defined"

    return probs

def get_cumulative_rewards(rewards,  # rewards at each step
                           gamma=0.99  # discount for reward
                           ):
    """
    Take a list of immediate rewards r(s,a) for the whole session
    and compute cumulative returns (a.k.a. G(s,a) in Sutton '16).

    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    A simple way to compute cumulative rewards is to iterate from the last
    to the first timestep and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    if len(rewards) == 0:
        return np.array([])
    
    cumulative_rewards = [0] * len(rewards)
    cumsum = 0
    for i in range(len(rewards) - 1, -1, -1):
        cumsum = rewards[i] + gamma * cumsum
        cumulative_rewards[i] = cumsum
        
    assert cumulative_rewards is not None, "cumulative_rewards is not defined"

    return np.array(cumulative_rewards)

def get_loss(states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    """
    Compute the loss for the REINFORCE algorithm.
    """
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int32)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    # predict logits, probas and log-probas using an agent.
    logits = model(states)
    assert logits is not None, "logits is not defined"

    probs = torch.softmax(logits, dim=1)
    assert probs is not None, "probs is not defined"

    log_probs = torch.log_softmax(logits, dim=1)
    assert log_probs is not None, "log_probs is not defined"

    assert all(isinstance(v, torch.Tensor) for v in [logits, probs, log_probs]), \
        "please use compute using torch tensors and don't use predict_probs function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    log_probs_for_actions = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    assert log_probs_for_actions is not None, "log_probs_for_actions is not defined"
    J_hat = torch.mean(log_probs_for_actions * cumulative_returns)
    assert J_hat is not None, "J_hat is not defined"
    
    # Compute loss here. Don't forget entropy regularization with `entropy_coef`
    entropy = -(probs * log_probs).sum(dim=1).mean()
    assert entropy is not None, "entropy is not defined"
    loss = -J_hat - entropy_coef * entropy
    assert loss is not None, "loss is not defined"

    return loss