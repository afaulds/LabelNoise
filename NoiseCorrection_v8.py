import math
from multiprocessing import Process, Manager
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
#from sklearn.cluster import KMeans
from faiss import Kmeans


class NoiseCorrection:
    """
    Implementation of original algorithm.
    """

    def __init__(self, X, y):
        self.M = 5 # Number of scores per item.
        self.K = 10 # Number of folds.
        self.C = 40
        self.X = X
        self.y = y
        self.r = None
        self.theta = 0
        # Thread safe objects.
        manager = Manager()
        self.Bc = manager.list(np.zeros(len(y)))
        self.Bx = manager.list(np.zeros(len(y)))
        self.noise_set = []

    def get_name():
        return "v8"

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

        # Calculate r (noise) and theta (threshold)
        self.r = np.zeros(len(self.y))
        self.theta = 0
        for i in range(len(self.y)):
            self.r[i] = self.Bx[i]
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
        kmeans = Kmeans(
            d=X_train.shape[1],
            k=self.C,
            niter = 300,
            min_points_per_centroid = 1,
            max_points_per_centroid = 10000000
        )
        kmeans.train(X_train.astype('float32'))
        D, I = kmeans.index.search(X_train.astype('float32'), 1)
        cluster_id = np.array(list(map(lambda x:x[0], I)))
        cluster_prob = np.zeros(self.C)
        for i in range(self.C):
            if np.sum(cluster_id == i) > 0:
                cluster_prob[i] = np.mean(y_train[cluster_id == i])
        D, I = kmeans.index.search(X_test.astype('float32'), 1)
        cluster_id = np.array(list(map(lambda x:x[0], I)))
        y_prob = np.array(list(map(lambda x:cluster_prob[x], cluster_id)))
        y_scores = y_prob > 0.5
        for p in range(len(test_index)):
            i = test_index[p]
            c = y_prob[p]
            d = 1.0 - c
            neg_entropy = 1.0
            if c > 0.0 and d > 0.0:
                neg_entropy = 1 + c * math.log2(c) + d * math.log2(d)
            if self.y[i] != y_scores[p]:
                self.Bc[i] += 1
                self.Bx[i] += 0.5 + 0.5 * neg_entropy
            else:
                self.Bx[i] += 0.5 - 0.5 * neg_entropy
