import numpy as np

from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    Theta1 = nn_params[0: hidden_layer_size*(input_layer_size+1)].\
    reshape((hidden_layer_size, input_layer_size+1))

    Theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].\
    reshape((num_labels, hidden_layer_size+1))



# Setup some useful variables
    m = np.size(X, 0)


# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m
#
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a 
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the 
#               first time.
#
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#



    # -------------------------------------------------------------

    # =========================================================================

    # Feed Forward
    a1 = np.concatenate((np.ones((m,1)), X), axis=1)
    z2 = a1.dot(Theta1.T);
    l2 = np.size(z2, 0);
    a2 = np.concatenate((np.ones((l2, 1)), sigmoid(z2)), axis=1)
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    yt = np.zeros((m, num_labels))
    yt[np.arange(m), y-1] = 1
    J = np.sum(-yt*np.log(a3)-(1-yt)*np.log(1-a3))

    # BackProp
    delta3 = a3-yt
    delta2 = delta3.dot(Theta2)*sigmoidGradient(np.concatenate((np.ones((l2, 1)), z2), axis=1))
    theta2_grad = delta3.T.dot(a2)
    theta1_grad = delta2[:, 1:].T.dot(a1)

    # Gradients
    J = J/m
    theta2_grad = theta2_grad/m
    theta2_grad[:, 1:] = theta2_grad[:, 1:]+ Lambda/m * Theta2[:, 1:]
    theta1_grad = theta1_grad/m
    theta1_grad[:, 1:] = theta1_grad[:, 1:]+Lambda/m*Theta1[:, 1:]
    reg_cost = np.sum(np.power(Theta1[:, 1:], 2)) + np.sum(np.power(Theta2[:, 1:], 2))
    J = J+1/(2*m)*Lambda*reg_cost

    # Unroll
    grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()))
    return J, grad