import numpy as np

class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance

def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    matches = []
    print(des1.shape, des2.shape)
    for i in range(len(des1)):
        dists1to2 = np.sqrt(np.sum((des1[i] - des2) ** 2, axis=1))
        best1to2 = dists1to2.argmin()
        dists2to1 = np.sqrt(np.sum((des2[best1to2] - des1) ** 2, axis=1))
        best2to1 = dists2to1.argmin()
        if best2to1 == i:
            matches.append(DummyMatch(i, best1to2, dists1to2[best1to2]))
    
    matches.sort(key=lambda x: x.distance)

    return matches
