import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from time import time
from NoiseCorrection import NoiseCorrection


def main():
    start_time = time()
    init()
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    g = []
    for i in range(30):
        score_perfect, score_dirty, score_nonoise50, score_nonoise75, score_nonoise100, score_nonoise125, score_clean = run_noise_removal()
        a.append(score_perfect)
        b.append(score_dirty)
        c.append(score_nonoise50)
        d.append(score_nonoise75)
        e.append(score_nonoise100)
        f.append(score_nonoise125)
        g.append(score_clean)
    print("Average perfect   {:.2f} {:.2f}".format(np.mean(a), np.std(a)))
    print("Average dirty     {:.2f} {:.2f}".format(np.mean(b), np.std(b)))
    print("Averagenonoise50  {:.2f} {:.2f}".format(np.mean(c), np.std(c)))
    print("Averagenonoise75  {:.2f} {:.2f}".format(np.mean(d), np.std(d)))
    print("Averagenonoise100 {:.2f} {:.2f}".format(np.mean(e), np.std(e)))
    print("Averagenonoise125 {:.2f} {:.2f}".format(np.mean(f), np.std(f)))
    print("Average clean     {:.2f} {:.2f}".format(np.mean(g), np.std(g)))
    end_time = time()
    print("Overall time: {}".format(end_time - start_time))


def init():
    if not os.path.exists('output'):
        os.makedirs('output')


def run_noise_removal():
    print("-------------------------------------")
    # Read from standardized file.
    with open("data/Simple2.pkl", "rb") as infile:
        data = pickle.loads(infile.read())

    # Create training and test set.
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"],
        data["y"],
        test_size=0.20
    )

    # Plot perfect
    plot(X_train, y_train, "output/step0.png")

    # Train and score if features were perfect.
    score_perfect = train_and_score(X_train, y_train, X_test, y_test)
    print("Perfect score {:.3f}".format(score_perfect))
    print("")

    # Randomize training set a little to introduce noise.
    changed = np.zeros(len(y_train))
    randomize(0.2, y_train, changed)
    print("Noise introduced wrong {}".format(np.sum(changed)))
    print("Training size {}".format(len(y_train)))

    # Plot noisy
    plot(X_train, y_train, "output/step1.png")

    # Noisy trained model.
    score_dirty = train_and_score(X_train, y_train, X_test, y_test)

    # Show score.
    print("Noisy score {:.3f}".format(score_dirty))
    print("")

    # Find noisy elements
    nc = NoiseCorrection(X_train, y_train)
    nc.calculate_noise()
    noise_set50 = nc.get_noise_index(0.5)
    noise_set75 = nc.get_noise_index(0.75)
    noise_set125 = nc.get_noise_index(1.25)
    noise_set100 = nc.get_noise_index()
    y_train_cleaned = nc.get_clean()

    # Remove noise 100
    all_index = range(len(y_train))
    good_index = np.setdiff1d(all_index, noise_set100)
    X_train_new = X_train[good_index]
    y_train_new = y_train[good_index]
    print("Data removed {}".format(len(noise_set100)))
    print("Noise removed {}".format(np.sum(changed[noise_set100])))
    print("Training size {}".format(len(y_train_new)))

    # Plot noise removed
    plot(X_train_new, y_train_new, "output/step2.png")

    # Train noise removed model.
    score_nonoise100 = train_and_score(X_train_new, y_train_new, X_test, y_test)
    # Show score.
    print("Noise removed 100% score {:.3f}".format(score_nonoise100))

    # Remove noise 75
    all_index = range(len(y_train))
    good_index = np.setdiff1d(all_index, noise_set75)
    X_train_new = X_train[good_index]
    y_train_new = y_train[good_index]
    score_nonoise75 = train_and_score(X_train_new, y_train_new, X_test, y_test)
    print("Noise removed 75% score {:.3f}".format(score_nonoise75))

    # Remove noise 125
    all_index = range(len(y_train))
    good_index = np.setdiff1d(all_index, noise_set125)
    X_train_new = X_train[good_index]
    y_train_new = y_train[good_index]
    score_nonoise125 = train_and_score(X_train_new, y_train_new, X_test, y_test)
    print("Noise removed 125% score {:.3f}".format(score_nonoise125))

    # Remove noise 50
    all_index = range(len(y_train))
    good_index = np.setdiff1d(all_index, noise_set50)
    X_train_new = X_train[good_index]
    y_train_new = y_train[good_index]
    score_nonoise50 = train_and_score(X_train_new, y_train_new, X_test, y_test)
    print("Noise removed 50% score {:.3f}".format(score_nonoise50))
    print("")

    print("Data changed {}".format(np.sum(y_train_cleaned != y_train)))
    print("Noise fixed {}".format(np.sum(np.logical_and(y_train_cleaned != y_train, changed == 1))))
    print("Training size {}".format(len(y_train_cleaned)))

    score_clean = train_and_score(X_train_new, y_train_new, X_test, y_test)
    print("Cleaned score {:.3f}".format(score_clean))

    # Plot noise corrected training
    plot(X_train, y_train_cleaned, "output/step3.png")

    return score_perfect, score_dirty, score_nonoise50, score_nonoise75, score_nonoise100, score_nonoise125, score_clean


def randomize(percent, y, changed):
    for i in range(len(y)):
        if random.random() < percent:
            y[i] = 1 - y[i]
            changed[i] = 1


def train_and_score(X_train, y_train, X_test, y_test):
    clf = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.8,
        max_depth=5
    ).fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)
    y_predict = np.zeros(len(y_scores))
    for i in range(len(y_scores)):
        y_predict[i] = y_scores[i][1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
    return metrics.roc_auc_score(y_test, y_predict)


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


if __name__ == "__main__":
    main()
