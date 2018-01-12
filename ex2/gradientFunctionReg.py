import numpy as np

from sigmoid import sigmoid
from gradientFunction import gradientFunction


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = np.size(y,0)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta


# =============================================================
    htheta = sigmoid(X.dot(theta))
    grad = np.zeros(np.size(theta, 0))
    grad[0] = 1/m * (X[:, 0].dot(htheta - y))
    grad[1:] = 1.0/m* ( X[:, 1:].T.dot(htheta - y)) + Lambda * theta[1:]/m 
    return grad