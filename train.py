import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import random
from NoiseCorrection import NoiseCorrection


def main():
    a = []
    b = []
    c = []
    d = []
    for i in range(30):
        score_perfect, score_dirty, score_nonoise, score_clean = test_noise_removal()
        a.append(score_perfect)
        b.append(score_dirty)
        c.append(score_nonoise)
        d.append(score_clean)
    print("Average perfect  {:.2f} {:.2f}".format(np.mean(a), np.std(a)))
    print("Average dirty    {:.2f} {:.2f}".format(np.mean(b), np.std(b)))
    print("Average no noise {:.2f} {:.2f}".format(np.mean(c), np.std(c)))
    print("Average clean    {:.2f} {:.2f}".format(np.mean(d), np.std(d)))


def test_noise_removal():
    print("-------------------------------------")
    # Read from standardized file.
    with open('data/Simple.pkl', 'rb') as infile:
        data = pickle.loads(infile.read())

    # Create training and test set.
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'],
        data['y'],
        test_size=0.20
    )

    plot(X_train, y_train, "output/step0.png")

    # Train and score if features were perfect.
    clf1 = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.8,
        max_depth=5
    ).fit(X_train, y_train)
    score_perfect = clf1.score(X_test, y_test)

    # Randomize training set a little to introduce noise.
    changed = np.zeros(len(y_train))
    randomize(0.2, y_train, changed)
    print("Noise introduced wrong {}".format(np.sum(changed)))
    print("Training size {}".format(len(y_train)))

    # Display so we can see noise.
    plot(X_train, y_train, "output/step1.png")

    # Train model.
    clf2 = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.8,
        max_depth=5
    ).fit(X_train, y_train)

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
    score_dirty = clf2.score(X_test, y_test)
    print("Training score {}".format(clf2.score(X_train, y_train)))
    print("Validation score {}".format(clf2.score(X_test, y_test)))
    print("")

    # Find noisy elements
    y_train_cleaned = y_train.copy()
    nc = NoiseCorrection(X_train, y_train)
    nc.calculate_noise()
    noise_set = nc.get_noise_index()

    # Remove noise
    all_index = range(len(y_train))
    good_index = np.setdiff1d(all_index, noise_set)
    X_train_new = X_train[good_index]
    y_train_new = y_train[good_index]
    print("Data removed {}".format(len(noise_set)))
    print("Noise removed {}".format(np.sum(changed[noise_set])))
    print("Training size {}".format(len(y_train_new)))

    plot(X_train_new, y_train_new, "output/step2.png")

    # Train model.
    clf3 = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.8,
        max_depth=5
    ).fit(X_train_new, y_train_new)

    # Predict and convert to single array.
    y_scores = clf3.predict_proba(X_test)
    y_predict = np.zeros(len(y_scores))
    for i in range(len(y_scores)):
        y_predict[i] = y_scores[i][1]

    # Get accuracy and precision curves.
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
    plt.plot(thresholds, fpr, 'r-', thresholds, tpr, 'g-')
    #plt.show()

    # Show score.
    score_nonoise = clf3.score(X_test, y_test)
    print("Training score {}".format(clf3.score(X_train_new, y_train_new)))
    print("Validation score {}".format(clf3.score(X_test, y_test)))
    print("")

    clf4 = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.8,
        max_depth=5
    ).fit(X_train, y_train_cleaned)
    score_clean = clf4.score(X_test, y_test)

    return score_perfect, score_dirty, score_nonoise, score_clean


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


if __name__ == '__main__':
    main()
