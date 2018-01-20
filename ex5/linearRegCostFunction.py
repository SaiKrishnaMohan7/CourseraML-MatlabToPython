import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = np.size(y, 0)# number of training examples
    grad = np.zeros(np.shape(theta))
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#

# =========================================================================
    J = 1/(2*m)*(X.dot(theta)-y).T.dot(X.dot(theta)-y) + Lambda/(2*m)*(theta[1:].dot(theta[1:]))
    grad[0] = 1/m*(X.dot(theta)-y).dot(X[:, 0])
    grad[1:] = 1/m*(X[:, 1:]).T.dot(X.dot(theta)-y) + Lambda/m*theta[1:]
    return J, grad