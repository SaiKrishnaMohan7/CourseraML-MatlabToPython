import numpy as np

from sigmoid import sigmoid


def lrGradFucntion(theta, X, y, Lambda):
    m = np.size(y, 0)
    htheta = sigmoid(X.dot(theta))
    grad = np.zeros(np.size(theta))
    grad[0] = 1/m * (X[:, 0].dot(htheta - y))
    grad[1:] = 1/m* ( X[:, 1:].T.dot(htheta - y)) + Lambda/m * theta[1:]

    return grad