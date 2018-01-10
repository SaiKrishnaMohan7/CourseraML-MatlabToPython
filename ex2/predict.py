import numpy as np

from sigmoid import sigmoid


def predict(theta, X):

    """ computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    m = np.size(X, 0)
    p = np.zeros((m, ))
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters.
#               You should set p to a vector of 0's and 1's
#

# =========================================================================
    pos = np.where(X.dot(theta) >= 0)
    neg = np.where(X.dot(theta) < 0)
    p[pos] = 1
    p[neg] = 0

    return p