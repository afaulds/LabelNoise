import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import math


def main():
    x = []
    y = []
    for i in range(10):
        score_orig, score_nonoise = test_noise_removal()
        x.append(score_orig)
        y.append(score_nonoise)

    print("Average base score {}".format(np.mean(x), ))
    print("Average no noise score {}".format(np.mean(y)))

def test_noise_removal():
    print("-------------------------------------")
    # Read from standardized file.
    with open('data/Simple2.pkl', 'rb') as infile:
        data = pickle.loads(infile.read())

    # Create training and test set.
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'],
        data['y'],
        test_size=0.20
    )

    plot(X_train, y_train, "output/step0.png")

    # Randomize training set a little to introduce noise.
    changed = np.zeros(len(y_train))
    randomize(0.25, y_train, changed)
    print("Noise introduced wrong {}".format(np.sum(changed)))
    print("Training size {}".format(len(y_train)))

    # Display so we can see noise.
    plot(X_train, y_train, "output/step1.png")

    # Train model.
    clf = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.8,
        max_depth=5
    ).fit(X_train, y_train)

    # Predict and convert to single array.
    y_scores = clf.predict_proba(X_test)
    y_predict = np.zeros(len(y_scores))
    for i in range(len(y_scores)):
        y_predict[i] = y_scores[i][1]

    # Get accuracy and precision curves.
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
    plt.plot(thresholds, fpr, 'r-', thresholds, tpr, 'g-')
    #plt.show()

    # Show score.
    score_base = clf.score(X_test, y_test)
    print("Training score {}".format(clf.score(X_train, y_train)))
    print("Validation score {}".format(clf.score(X_test, y_test)))
    print("")

    all_index = range(len(y_train))
    noise = find_noise(X_train, y_train)
    good_index = np.setdiff1d(all_index, noise)
    X_train_new = X_train[good_index]
    y_train_new = y_train[good_index]
    print("Data removed {}".format(len(noise)))
    print("Noise removed {}".format(np.sum(changed[noise])))
    print("Training size {}".format(len(y_train_new)))

    plot(X_train_new, y_train_new, "output/step2.png")

    # Train model.
    clf2 = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.8,
        max_depth=5
    ).fit(X_train_new, y_train_new)

    # Predict and convert to single array.
    y_scores = clf2.predict_proba(X_test)
    y_predict = np.zeros(len(y_scores))
    for i in range(len(y_scores)):
        y_predict[i] = y_scores[i][1]

    # Get accuracy and precision curves.
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
    plt.plot(thresholds, fpr, 'r-', thresholds, tpr, 'g-')
    #plt.show()

    # Show score.
    score_nonoise = clf2.score(X_test, y_test)
    print("Training score {}".format(clf2.score(X_train_new, y_train_new)))
    print("Validation score {}".format(clf2.score(X_test, y_test)))
    print("")

    return score_base, score_nonoise


def randomize(percent, y, changed):
    for i in range(len(y)):
        if random.random() < percent:
            y[i] = 1 - y[i]
            changed[i] = 1


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
    plt.plot(x1, y1, 'r^', x2, y2, 'bs')
    plt.savefig(file_name)
    plt.close()


def find_noise(X, y):
    m = 5 # Number of scores per item.
    k = 10 # Number of folds.
    y_predicts = np.zeros((len(y), m))
    Bc = np.zeros(len(y))
    Be = np.zeros(len(y))
    all_index = range(len(y))
    Be_total = 0
    for j in range(m):
        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = GradientBoostingClassifier(
                n_estimators=10,
                learning_rate=0.8,
                max_depth=5
            ).fit(X_train, y_train)
            y_scores = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            for p in range(len(test_index)):
                i = test_index[p]
                c = y_prob[p][0]
                d = y_prob[p][1]
                if c > 0.0 and d > 0.0:
                    Be[i] -= c * math.log2(c) + d * math.log2(d)
                Be_total += Be[i]
                if y[i] != y_scores[p]:
                    Bc[i] += 1

    r = np.zeros(len(y))
    theta = 0
    for i in range(len(y)):
        r[i] = Bc[i] + (1 - Be[i] / Be_total)
        if Bc[i] >= m / 2:
            theta += 1
    sorted_index = [x for _,x in sorted(zip(r, all_index), reverse=True)]
    return sorted_index[0:theta]


if __name__ == '__main__':
    main()
