import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m = np.size(X, 0)
    X = np.concatenate((np.ones((m, 1)), X), axis = 1)

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#

# =========================================================================
    temp1 = sigmoid(X.dot(Theta1.T))
    temp = np.concatenate((np.ones((m, 1)), temp1), axis = 1)
    temp2 = sigmoid(temp.dot(Theta2.T))
    p = np.argmax(temp2, axis=1) + 1
    return p

