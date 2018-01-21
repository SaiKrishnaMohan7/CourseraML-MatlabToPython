import matplotlib.pyplot as plt
import numpy as np


def drawLine(p1, p2, lc='k-', lwidth=2):
    x = np.array([p1[0], p2[0]])
    y = np.array([p1[1], p2[1]])
    plt.plot(x, y, lc, lw=lwidth)
    plt.show()