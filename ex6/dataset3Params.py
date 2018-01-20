import numpy as np
import sklearn.svm


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

# You need to return the following variables correctly.
    C = 1
    sigma = 0.3
    err_best = np.size(yval, 0)

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the optimal C and sigma
#               learning parameters found using the cross validation set.
#               You can use svmPredict to predict the labels on the cross
#               validation set. For example, 
#                   predictions = svmPredict(model, Xval)
#               will return the predictions on the cross validation set.
#
#  Note: You can compute the prediction error using 
#        mean(double(predictions ~= yval))
#
    C_choice = [0.3, 0.1]
    sigma_choice = [0.1, 0.3]

    for i in range(len(C_choice)):
        for j in range(len(sigma_choice)):
            clf = svm.SVC(C=c_choice[i], gamma=1/(2*sigma_choice[j]**2))
            clf.fit(X,y)
            pred = clf.predict(Xval)
            err = np.sum(pred != yval)/err_best
            if err_best > 0:
                err_best = err
                c = C_choice[i]
                sigma = sigma_choice[j]

# =========================================================================
    return C, sigma
