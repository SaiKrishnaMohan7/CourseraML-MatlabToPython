import numpy as np
import math

def selectThreshold(yval, pval):
    """
    finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    bestEpsilon = 0
    bestF1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000.0
    ietrable = np.arange(np.min(pval), np.max(pval), stepsize).tolist()
    for epsilon in ietrable:

        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions

        tp = np.sum(np.logical_and(pval < epsilon, yval == 1))
        fp = np.sum(np.logical_and(pval < epsilon, yval == 0))
        fn = np.sum(np.logical_and(pval >= epsilon, yval == 1))
        if tp+fp == 0 or tp+fn == 0:
            F1 = -1
        else:
            prec = tp/(tp+fp)
            rec = tp/(tp+fn)
            F1 = 2*prec*rec/(prec+rec)
        # =============================================================

        if F1 > bestF1:
           bestF1 = F1
           bestEpsilon = epsilon

    return bestEpsilon, bestF1






