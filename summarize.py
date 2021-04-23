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
from StatCompare import StatCompare
from util import Cache
from util import Grid


num_repeat_runs = 40
input_data_files = [
    "data/Biodeg.pkl",
    "data/Ionosphere.pkl",
    "data/Krvskp.pkl",
    "data/Mushroom.pkl",
#    "data/Sick.pkl",
    "data/Simple.pkl",
    "data/Simple2.pkl",
    "data/Spam.pkl",
    "data/Tictactoe.pkl",
    "data/Unbalanced.pkl",
    "data/Vote.pkl",
]
noise_types = [
    v0.NoiseCorrection.get_name(),
    v1.NoiseCorrection.get_name(),
    v2.NoiseCorrection.get_name(),
    v3.NoiseCorrection.get_name(),
    v4.NoiseCorrection.get_name(),
    v5.NoiseCorrection.get_name(),
    v6.NoiseCorrection.get_name(),
    v7.NoiseCorrection.get_name(),
]
stat_key = "auc"


def main():
    start_time = time()
    grid = Grid()
    grid.load("test.txt")
    for file_name in input_data_files:
        for noise_name in noise_types:
            scores = []
            for i in range(num_repeat_runs):
                key = (i, noise_name, file_name)
                score = Cache.get(key)
                if score is not None:
                    scores.append(score)
                else:
                    print("Missing - {}".format(key))
            val = StatCompare.mean(scores)
            if stat_key in val:
                val = val[stat_key]
            else:
                val = ""
            grid.set(file_name, noise_name, val)
    grid.save("test.txt")
    end_time = time()
    print("Stats for {}".format(stat_key))
    print("Overall time: {}".format(end_time - start_time))


if __name__ == "__main__":
    main()
