import numpy as np

from costFunction import costFunction
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = np.size(y, 0)  # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

# =============================================================
    htheta = sigmoid(X.dot(theta))
    J = -1/m*(y.dot(np.log(htheta))+(1-y).dot(np.log(1-htheta))) + Lambda/(2*m)*theta[1:].dot(theta[1:])
    return J
