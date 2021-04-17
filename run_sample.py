import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import os
from time import time
from NoiseCorrection_v0 as v0
from NoiseCorrection_v1 as v1
import helper


num_repeat_runs = 30
input_data = "data/Unbalanced.pkl"
noise_percent = 0.1


def main():
    scores = run_noise_removal()
    print(scores)


def init():
    if not os.path.exists('output'):
        os.makedirs('output')


def run_noise_removal():
    print("-------------------------------------")
    # Read from standardized file.
    with open(input_data, "rb") as infile:
        data = pickle.loads(infile.read())

    # Create training and test set.
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"],
        data["y"],
        test_size=0.20
    )

    # Plot perfect
    helper.plot(X_train, y_train, "output/step0.png")

    # Train and score if features were perfect.
    score_perfect = train_and_score(X_train, y_train, X_test, y_test)
    print("Perfect score {:.3f}".format(score_perfect))
    print("")

    # Randomize training set a little to introduce noise.
    changed = helper.randomize(noise_percent, y_train)
    print("Noise introduced into training set {}".format(np.sum(changed)))
    print("Training size {}".format(len(y_train)))

    # Plot noisy
    helper.plot(X_train, y_train, "output/step1.png")

    # Find noisy elements
    nc = v0.NoiseCorrection(X_train, y_train)
    nc.calculate_noise()
    noise_set0 = nc.get_noise_set(0.0)
    noise_set25 = nc.get_noise_set(0.25)
    noise_set50 = nc.get_noise_set(0.5)
    noise_set75 = nc.get_noise_set(0.75)
    noise_set100 = nc.get_noise_set()
    noise_set125 = nc.get_noise_set(1.25)
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
    helper.plot(X_train_new, y_train_new, "output/step2.png")

    score_no_noise_0 = evaluate_score(X_train, y_train, X_test, y_test, noise_set0)
    score_no_noise_25 = evaluate_score(X_train, y_train, X_test, y_test, noise_set25)
    score_no_noise_50 = evaluate_score(X_train, y_train, X_test, y_test, noise_set50)
    score_no_noise_75 = evaluate_score(X_train, y_train, X_test, y_test, noise_set75)
    score_no_noise_100 = evaluate_score(X_train, y_train, X_test, y_test, noise_set100)
    score_no_noise_125 = evaluate_score(X_train, y_train, X_test, y_test, noise_set125)

    print("Data changed {}".format(np.sum(y_train_cleaned != y_train)))
    print("Noise fixed {}".format(np.sum(np.logical_and(y_train_cleaned != y_train, changed == 1))))
    print("Training size {}".format(len(y_train_cleaned)))

    score_clean = train_and_score(X_train_new, y_train_new, X_test, y_test)
    print("Cleaned score {:.3f}".format(score_clean))

    # Plot noise corrected training
    helper.plot(X_train, y_train_cleaned, "output/step3.png")

    return {
        "perfect": score_perfect,
        "clean": score_clean,
        "no_noise_0": score_no_noise_0,
        "no_noise_25": score_no_noise_25,
        "no_noise_50": score_no_noise_50,
        "no_noise_75": score_no_noise_75,
        "no_noise_100": score_no_noise_100,
        "no_noise_125": score_no_noise_125
    }


def evaluate_score(X_train, y_train, X_test, y_test, noise_set):
    all_index = range(len(y_train))
    good_index = np.setdiff1d(all_index, noise_set)
    X_train_new = X_train[good_index]
    y_train_new = y_train[good_index]
    score_no_noise = train_and_score(X_train_new, y_train_new, X_test, y_test)
    print("Noise removed score {:.3f}".format(score_no_noise))
    return score_no_noise


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
    return metrics.auc(fpr, tpr)


if __name__ == "__main__":
    main()
