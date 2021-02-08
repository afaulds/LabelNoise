import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import random


def main():
    with open('data/Simple.pkl', 'rb') as infile:
        data = pickle.loads(infile.read())


    X_train, X_test, y_train, y_test = train_test_split(
        data['X'],
        data['y'],
        test_size=0.20
    )

    randomize(y_train)

    clf = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.8,
        max_depth=5
    ).fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)
    y_predict = np.zeros(len(y_scores))
    for i in range(len(y_scores)):
        y_predict[i] = y_scores[i][1]

    print(np.round(y_predict, 2))
    print(np.round(y_test, 2))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
    plt.plot(thresholds, fpr, 'r-', thresholds, tpr, 'g-')
    #plt.show()

    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))


def randomize(y):
    for i in range(len(y)):
        if random.random() < 0.2:
            y[i] = 1 - y[i]


if __name__ == '__main__':
    main()
