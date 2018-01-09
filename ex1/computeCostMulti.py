import numpy as np

def computeCostMulti(X, y, theta):
    """
     Compute cost for linear regression with multiple variables
       J = computeCost(X, y, theta) computes the cost of using theta as the
       parameter for linear regression to fit the data points in X and y
    """
    m = np.size(y, 0)
    J = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.

    htheta = X.T.dot(X,theta)
    J = np.sum (1.0/(2*m) * (np.square(htheta - y)))
# =========================================================================

    return J
