import numpy as np

from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    theta_history = []
    J_history = []
    m = np.size(y, 0)  # number of training examples

    for i in range(num_iters):
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #



        # ============================================================
        htheta = X.T.dot(X, theta)
        theta = theta - alpha*(1.0/m)*np.dot(X.T, htheta-y)  
        # Save the cost J in every iteration
        theta_history.append(theta) 
        J_history[i] = computeCostMulti(X, y, htheta)
        
    return theta, theta_history, J_history
