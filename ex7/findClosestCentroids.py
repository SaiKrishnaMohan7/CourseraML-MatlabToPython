import numpy as np


def findClosestCentroids(X, centroids):
    """returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

# Set K
    K, l = len(centroids)

# You need to return the following variables correctly.
    idx = np.zeros(X.shape[0])
    xtemp = np.tile(x,k)

# ====================== YOUR CODE HERE ======================
# Instructions: Go over every example, find its closest centroid, and store
#               the index inside idx at the appropriate location.
#               Concretely, idx(i) should contain the index of the centroid
#               closest to example i. Hence, it should be a value in the 
#               range 1..K
#
# Note: You can use a for-loop over the examples to compute this.


# =============================================================
    centeroid_temp = centroids.flatten()
    xtemp = np.power(xtemp-centeroid_temp, 2)
    xt = np.zeros((np.size(xtemp, 0), K))
    for i in range(K):
        for j in range(l):
            xt[:, i] = xt[:, i] + xtemp[:, i*l+j]
    idx = np.argmin(xt, 1) + 1

    return idx

