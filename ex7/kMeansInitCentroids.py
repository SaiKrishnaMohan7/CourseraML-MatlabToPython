import numpy as np


def kMeansInitCentroids(X, K):
    """returns K initial centroids to be
    used with the K-Means on the dataset X
    """


# ====================== YOUR CODE HERE ======================
# Instructions: You should set centroids to randomly chosen examples from
#               the dataset X
#

    randidx = np.random.permutation(np.size(X, 0))
    centroids = X[randidx[0: K], :]
# =============================================================
    print('kMeansInit----->', centroids);
    return centroids
