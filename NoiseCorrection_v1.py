import math
from multiprocessing import Process, Manager
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold


class NoiseCorrection:
    """
    Implementation of original algorithm.
    """

    def __init__(self, X, y):
        self.M = 5 # Number of scores per item.
        self.K = 10 # Number of folds.
        self.X = X
        self.y = y
        self.r = None
        self.theta = 0
        # Thread safe objects.
        manager = Manager()
        self.Bc = manager.list(np.zeros(len(y)))
        self.Be = manager.list(np.zeros(len(y)))
        self.noise_set = []

    def get_name():
        return "v1"

    def calculate_noise(self):
        jobs = []
        # Run all k-fold models in parallel.
        for m in range(self.M):
            kf = KFold(n_splits=self.K)
            for train_index, test_index in kf.split(self.X):
                p = Process(target=self.train_and_eval, args=(m, train_index, test_index))
                p.start()
                jobs.append(p)
                print(".", end="", flush=True)
        print("All started.", flush=True)

        # Wait for jobs to complete.
        for job in jobs:
            job.join()
            print("*", end="", flush=True)
        print("All complete.", flush=True)

        # Calculate Be_total
        Be_total = np.sum(self.Be)

        # Calculate r (noise) and theta (threshold)
        self.r = np.zeros(len(self.y))
        self.theta = 0
        for i in range(len(self.y)):
            self.r[i] = self.Bc[i] + (1 - self.Be[i] / Be_total)
            if self.Bc[i] >= self.M / 2:
                self.theta += 1

    def get_noise_score(self):
        return self.r

    def get_noise_set(self, fracion=1.0):
        all_index = range(len(self.y))
        sorted_index = [x for _,x in sorted(zip(self.r, all_index), reverse=True)]
        self.noise_set = sorted_index[0:int(self.theta * fracion)]
        return self.noise_set

    def train_and_eval(self, m, train_index, test_index):
        X_train = self.X[train_index]
        y_train = self.y[train_index]
        X_test = self.X[test_index]
        y_test = self.y[test_index]
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
                self.Be[i] -= c * math.log2(c) + d * math.log2(d)
            if self.y[i] != y_scores[p]:
                self.Bc[i] += 1
