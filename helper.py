import matplotlib.pyplot as plt1
import numpy as np
import random
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import fbeta_score


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


def evaluate_score(X_train, y_train, X_test, y_test, noise_set):
    all_index = range(len(y_train))
    good_index = np.setdiff1d(all_index, noise_set)
    X_train_new = X_train[good_index]
    y_train_new = y_train[good_index]
    score_no_noise = train_and_score(X_train_new, y_train_new, X_test, y_test)
    return score_no_noise


def train_and_score(X_train, y_train, X_test, y_test):
    if len(np.unique(y_train)) > 1:
        clf = GradientBoostingClassifier(
            n_estimators=10,
            learning_rate=0.8,
            max_depth=5
        ).fit(X_train, y_train)
        y_scores = clf.predict_proba(X_test)
        y_predict = np.zeros(len(y_scores))
        for i in range(len(y_scores)):
            y_predict[i] = y_scores[i][1]
    else:
        print("EVALUATE ERROR")
        y_scores = np.ones(len(y_test)) * np.unique(y_train)[0]
        y_predict = y_scores
    return fbeta_score(y_test, y_predict)
