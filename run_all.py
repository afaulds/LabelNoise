import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import os
from time import time
import NoiseCorrection_v0 as v0
import NoiseCorrection_v1 as v1
import helper


num_repeat_runs = 1 #30
input_data_files = ["data/Simple.pkl"]
[
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
noise_classes = {
    "v0": v0.NoiseCorrection,
    "v1": v1.NoiseCorrection
}
noise_percent = 0.2


def main():
    init()
    start_time = time()
    scores_agg = {}
    for file_name in input_data_files:
        print("Process {}...".format(file_name))
        for noise_name in noise_classes:
            print("Process {}...".format(noise_name))
            noise_class = noise_classes[noise_name]
            if noise_name not in scores_agg:
                scores_agg[noise_name] = {}
            for i in range(num_repeat_runs):
                scores = run_noise_removal(file_name, noise_class)
                for key in scores:
                    if key not in scores_agg[noise_name]:
                        scores_agg[noise_name][key] = []
                    scores_agg[noise_name][key].append(scores[key])
    print(scores_agg)
    #        for key in scores_agg:
    #            print("Average {}   {:.2f} {:.2f}".format(key, np.mean(scores_agg[key]), np.std(scores_agg[key])))
    #        end_time = time()
    #        print("Overall time: {}".format(end_time - start_time))


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
        "no_noise_125": score_no_noise_125
    }


if __name__ == "__main__":
    main()
