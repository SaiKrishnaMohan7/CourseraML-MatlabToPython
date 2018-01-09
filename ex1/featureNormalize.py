import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    X_norm, mu, sigma = 0,0,0
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'mean' and 'std' functions useful.
    #

    stored_feature_means, stored_feature_stds = [], []
    Xnorm = X.copy()
    for icol in range(Xnorm.shape[1]):
        stored_feature_means.append(np.mean(Xnorm[:,icol]))
        stored_feature_stds.append(np.std(Xnorm[:,icol]))
        #Skip the first column
        if not icol: continue
        #Faster to not recompute the mean and std again, just used stored values
        Xnorm[:,icol] = (Xnorm[:,icol] - stored_feature_means[-1])/stored_feature_stds[-1]
# ============================================================

    return Xnorm, stored_feature_means, stored_feature_stds 
