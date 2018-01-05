import matplotlib.pyplot as plt
import numpy as np

def plotData(X, y):
    """
    plots the data points and gives the figure axes labels of
    population and profit.
    """

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the
#               "figure" and "plot" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the
#               population and revenue data have been passed in
#               as the x and y arguments of this function.
#
# Hint: You can use the 'rx' option with plot to have the markers
#       appear as red crosses. Furthermore, you can make the
#       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

    plt.figure(figsize=(10, 6))  # open a new figure window
    plt.plot(X, y, 'rx', ms = 10)
    plt.xlabel('Population of Cities in 10,000')
    plt.ylabel('Profit in $10,000')
    plt.show()

# ============================================================

# Plot the convergence of the cost function to check if
# the algorithm is working as intended
def plotConvergence(J_history, num_iter):
    plt.figure(figsize=(10, 6))
    # len(J_history) will give us the number of iterations
    plt.plot(range(len(J_history)), J_history, 'bo')
    plt.grid(True)
    plt.title('Convergence of Cost Function')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost Function')
    dummy = plt.xlim([-0.05 * num_iter, 1.05 * num_iter])
