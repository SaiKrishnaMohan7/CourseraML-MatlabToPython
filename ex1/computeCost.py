import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = np.size(y, 0)
    J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
    J = (X.dot(theta)-y).dot(X.dot(theta)-y)/(2*m)

# =========================================================================

    return J


