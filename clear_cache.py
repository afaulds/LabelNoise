import os
from time import time
from util import Cache
import NoiseCorrection_v0 as v0
import NoiseCorrection_v1 as v1
import NoiseCorrection_v2 as v2
import NoiseCorrection_v3 as v3
import NoiseCorrection_v4 as v4

num_repeat_runs = 20
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
    "data/Vote.pkl",
]
noise_types = [
#    v0.NoiseCorrection.get_name(),
#    v1.NoiseCorrection.get_name(),
#    v2.NoiseCorrection.get_name(),
    v3.NoiseCorrection.get_name(),
    v4.NoiseCorrection.get_name(),
]


def main():
    start_time = time()
    for file_name in input_data_files:
        for noise_name in noise_types:
            scores = []
            for i in range(num_repeat_runs):
                key = (i, noise_name, file_name)
                score = Cache.delete(key)
    end_time = time()
    print("Overall time: {}".format(end_time - start_time))


if __name__ == "__main__":
    main()
