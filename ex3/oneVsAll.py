import numpy as np
import scipy.optimize as op

from lrCostFunction import lrCostFunction
from lrGradFunction import lrGradFucntion


def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

# Some useful variables
    m, n = np.shape(X)

# You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

# Add ones to the X data matrix
    X = np.concatenate((np.ones((m,1)), X), axis = 1)

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda. 
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

    # Set Initial theta
    

    for i in range(num_labels):
        num = 10 if i == 0 else i
        initial_theta = np.zeros((n + 1,))
        result = op.minimize(lrCostFunction, initial_theta, method='BFGS'\
                    , jac = lrGradFucntion, args=(X, 1*(y == num), Lambda), options = {'maxiter': 50})



# =========================================================================

    return all_theta

