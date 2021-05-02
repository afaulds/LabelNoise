import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import os
from time import time
import NoiseCorrection_v0 as v0
import NoiseCorrection_v1 as v1
import NoiseCorrection_v2 as v2
import NoiseCorrection_v3 as v3
import NoiseCorrection_v4 as v4
import NoiseCorrection_v5 as v5
import NoiseCorrection_v6 as v6
import NoiseCorrection_v7 as v7
import NoiseCorrection_v8 as v8
import NoiseCorrection_v9 as v9
import helper
import random
from sklearn import metrics
from StatCompare import StatCompare
from util import Cache
import pprint


num_repeat_runs = 40
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
noise_classes = [
    v0.NoiseCorrection,
    v1.NoiseCorrection,
    v2.NoiseCorrection,
#    v3.NoiseCorrection,
#    v4.NoiseCorrection,
#    v5.NoiseCorrection,
#    v6.NoiseCorrection,
#    v7.NoiseCorrection,
    v8.NoiseCorrection,
    v9.NoiseCorrection,
]
noise_percent = 0.2


def main():
    init()
    start_time = time()
    z = 0
    total = num_repeat_runs * len(input_data_files) * len(noise_classes)
    for file_name in input_data_files:
        print("Process {}...".format(file_name))

        for noise_class in noise_classes:
            print("Process {}...".format(noise_class.get_name()))
            for i in range(num_repeat_runs):
                print("Process [{}/{}]...".format(z + 1, total))

                key = (i, noise_class.get_name(), file_name)
                score = Cache.process(key, run_noise_removal, file_name, noise_class)
                print(score['auc'])
                z += 1

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


if __name__ == "__main__":
    main()
