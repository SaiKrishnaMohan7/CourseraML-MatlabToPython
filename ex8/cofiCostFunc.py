import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    """returns the cost and gradient for the
    """

    # Unfold the U and W matrices from params
    X = np.array(params[:num_movies*num_features]).reshape(num_features, num_movies).T.copy()
    Theta = np.array(params[num_movies*num_features:]).reshape(num_features, num_users).T.copy()


    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the
    #                     partial derivatives w.r.t. to each element of Theta


    # =============================================================
    J = 1/2*np.sum(R*(X.dot(Theta.T)-Y)**2)+Lambda/2*(np.sum(Theta**2)+np.sum(X**2))
    
    for i in range(np.size(X, 0)):
        idx = R[i, :] == 1
        X_grad[i, :] = (X[i, :].dot(Theta[idx, :].T)-Y[i, idx]).dot(Theta[idx, :])+Lambda*X[i, :]
    
    for j in range(np.size(Theta, 0)):
        jdx = R[:, j] == 1
        Theta_grad[j, :] = (Theta[j, :].dot(X[jdx, :].T)-Y[jdx, j].T).dot(X[jdx, :])+Lambda*Theta[j, :]
    
    grad = np.hstack((X_grad.T.flatten(),Theta_grad.T.flatten()))
    return J, grad
