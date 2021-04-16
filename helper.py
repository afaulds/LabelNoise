import numpy as np
import random
import matplotlib.pyplot as plt


def randomize(percent, y):
    """
    Given a set y, and a percent to update,
    randomly change labels for percent of them
    to the wrong label. Update the original y
    set and indicate in the changed set which
    ones were updated.

    Args:
        percent (int) - Percent to change to wrong label.
        y (int array) - Original label array.
        changed (int array) - Array of ones (items that were changed) and zeros (those that are left alone)
    """
    changed = np.zeros(len(y))
    for i in range(len(y)):
        if random.random() < percent:
            y[i] = 1 - y[i]
            changed[i] = 1
    return changed


def plot(X, y, file_name):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(len(y)):
        if y[i] == 0:
            x1.append(X[i][0])
            y1.append(X[i][1])
        else:
            x2.append(X[i][0])
            y2.append(X[i][1])
    plt.plot(x1, y1, "r^", x2, y2, "bs")
    plt.savefig(file_name)
    plt.close()
