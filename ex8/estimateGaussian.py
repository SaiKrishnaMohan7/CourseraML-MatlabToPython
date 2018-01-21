import numpy as np


def estimateGaussian(X):
    """
    This function estimates the parameters of a
    Gaussian distribution using the data in X
      The input X is the dataset with each n-dimensional data point in one row
      The output is an n-dimensional vector mu, the mean of the data set
      and the variances sigma^2, an n x 1 vector
    """
    m, n = X.shape

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the mean of the data and the variances
    #               In particular, mu(i) should contain the mean of
    #               the data for the i-th feature and sigma2(i)
    #               should contain variance of the i-th feature.
    #

    mu = np.sum(X,0)/m
    sigma = np.sum(np.power(X-mu, 2), 0)/m
# =============================================================

    return mu, sigma


