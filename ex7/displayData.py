import numpy as np
import math
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

from show import show

def displayData(X):
    """displays 2D data
      stored in X in a nice grid. It returns the figure handle h and the
      displayed array if requested."""

    width = round(math.sqrt(np.size(X, 1)))
    m, n = np.shape(X)
    height = int(n/width)

    drows = math.floor(math.sqrt(m))
    dcols = math.ceil(m/drows)

    pad = 1

    darray = -1*np.ones((pad+drows*(height+pad), pad+dcols*(width+pad)))

    curr_ex = 0
    for j in range(drows):
        for i in range(dcols):
            if curr_ex >= m:
                break
            max_val = np.max(np.abs(X[curr_ex, :]))
            darray[pad+j*(height+pad):pad+j*(height+pad)+height, pad+i*(width+pad):pad+i*(width+pad)+width]\
                = X[curr_ex, :].reshape((height, width))/max_val
            curr_ex += 1
        if curr_ex >= m:
            break

    plt.imshow(darray.T, cmap='gray')
    plt.show()



