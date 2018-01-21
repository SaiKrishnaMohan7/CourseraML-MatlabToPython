import math
import numpy as np
import scipy.linalg as la


def multivariateGaussian(X, mu, Sigma2):
    """Computes the probability
    density function of the examples X under the multivariate gaussian
    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    treated as the covariance matrix. If Sigma2 is a vector, it is treated
    as the \sigma^2 values of the variances in each dimension (a diagonal
    covariance matrix)
    """
    k = np.size(mu, 0)
    Sigma2 = np.diag(Sigma2)
    X = X - mu
    p = (2*math.pi)**(-k/2)*la.det(Sigma2)**(-0.5)*np.exp(-0.5*np.sum(X.dot(la.pinv(Sigma2))*X, 1))

    return p