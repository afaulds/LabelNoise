import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import os
from time import time
import NoiseCorrection_v0 as v0
import NoiseCorrection_v1 as v1
import helper
import random
from sklearn import metrics
from StatCompare import StatCompare


num_repeat_runs = 10
input_data_files = [
    "data/Biodeg.pkl",
    "data/Ionosphere.pkl",
    "data/Krvskp.pkl",
    "data/Mushroom.pkl",
    "data/Sick.pkl",
    "data/Simple.pkl",
    "data/Simple2.pkl",
    "data/Spam.pkl",
    "data/Tictactoe.pkl",
    "data/Unbalanced.pkl",
    "data/Vote.pkl"
]
class_a = v0.NoiseCorrection
class_b = v1.NoiseCorrection
noise_percent = 0.2


def main():
    init()
    start_time = time()
    for file_name in input_data_files:
        print("Process {}...".format(file_name))

        scores_a = []
        scores_b = []

        print("Process {}...".format(class_a.get_name()))
        for i in range(num_repeat_runs):
            scores_a.append(run_noise_removal(file_name, class_a))

        print("Process {}...".format(class_b.get_name()))
        for i in range(num_repeat_runs):
            scores_b.append(run_noise_removal(file_name, class_b))

        print(StatCompare.diff(scores_a, scores_b))
    end_time = time()
    print("Overall time: {}".format(end_time - start_time))


def init():
    if not os.path.exists('output'):
        os.makedirs('output')


def run_noise_removal(file_name, noise_class):
    # Read from standardized file.
    with open(file_name, "rb") as infile:
        data = pickle.loads(infile.read())

    # Create training and test set.
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"],
        data["y"],
        test_size=0.20
    )

    # Train and score if features were perfect.
    score_perfect = helper.evaluate_score(X_train, y_train, X_test, y_test, [])

    # Randomize training set a little to introduce noise.
    changed = helper.randomize(noise_percent, y_train)

    # Train and score if noise was perfectly removed.
    all_index = np.array(range(len(y_train)))
    noise_set = all_index[changed == 1]
    score_no_noise_perfect = helper.evaluate_score(X_train, y_train, X_test, y_test, noise_set)

    # Find noisy elements
    nc = noise_class(X_train, y_train)
    nc.calculate_noise()
    noise_score = nc.get_noise_score()
    fpr, tpr, thresholds = metrics.roc_curve(changed, noise_score)
    auc = metrics.auc(fpr, tpr)

    noise_set0 = nc.get_noise_set(0.0)
    noise_set25 = nc.get_noise_set(0.25)
    noise_set50 = nc.get_noise_set(0.5)
    noise_set75 = nc.get_noise_set(0.75)
    noise_set100 = nc.get_noise_set()
    noise_set125 = nc.get_noise_set(1.25)

    score_no_noise_0 = helper.evaluate_score(X_train, y_train, X_test, y_test, noise_set0)
    score_no_noise_25 = helper.evaluate_score(X_train, y_train, X_test, y_test, noise_set25)
    score_no_noise_50 = helper.evaluate_score(X_train, y_train, X_test, y_test, noise_set50)
    score_no_noise_75 = helper.evaluate_score(X_train, y_train, X_test, y_test, noise_set75)
    score_no_noise_100 = helper.evaluate_score(X_train, y_train, X_test, y_test, noise_set100)
    score_no_noise_125 = helper.evaluate_score(X_train, y_train, X_test, y_test, noise_set125)

    return {
        "perfect": score_perfect,
        "no_noise_perfect": score_no_noise_perfect,
        "no_noise_0": score_no_noise_0,
        "no_noise_25": score_no_noise_25,
        "no_noise_50": score_no_noise_50,
        "no_noise_75": score_no_noise_75,
        "no_noise_100": score_no_noise_100,
        "no_noise_125": score_no_noise_125,
        "auc": auc
    }


def clear_stats():
    pass


def write_stats():
    pass


def write_results(scores_agg):
    print(scores_agg)
    zscore, prob = ttest_ind(a_dist, b_dist, equal_var=False)
    print(f"Zscore is {zscore:0.2f}, p-value is {prob:0.3f} (two tailed), {prob/2:0.3f} (one tailed)")


if __name__ == "__main__":
    main()
