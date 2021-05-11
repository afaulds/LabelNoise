import math
from multiprocessing import Process, Manager
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import cleanlab


class NoiseCorrection:
    """
    Implementation of original algorithm.
    """

    def __init__(self, X, y):
        self.M = 5 # Number of scores per item.
        self.K = 10 # Number of folds.
        self.C = 10
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
        return "v10"

    def calculate_noise(self):
        s = self.y.astype(int)
        psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
            self.X, s, clf=GradientBoostingClassifier(n_estimators=10,
            learning_rate=0.8,
            max_depth=5
        ))
        s = np.asarray(s)
        psx = np.asarray(psx)

        # Find the number of unique classes if K is not given
        K = len(np.unique(s))

        thresholds = [np.mean(psx[:,k][s == k]) for k in range(K)] # P(s^=k|s=k)
        thresholds = np.asarray(thresholds)

        confident_joint = np.zeros((K, K), dtype = int)
        for i, row in enumerate(psx):
            s_label = s[i]
            # Find out how many classes each example is confidently labeled as
            confident_bins = row >= thresholds - 1e-6
            num_confident_bins = sum(confident_bins)
            # If more than one conf class, inc the count of the max prob class
            if num_confident_bins == 1:
                confident_joint[s_label][np.argmax(confident_bins)] += 1
            elif num_confident_bins > 1:
                confident_joint[s_label][np.argmax(row)] += 1

        # Normalize confident joint (use cleanlab, trust me on this)
        confident_joint = cleanlab.latent_estimation.calibrate_confident_joint(
            confident_joint, s)

        MIN_NUM_PER_CLASS = 5
        # Leave at least MIN_NUM_PER_CLASS examples per class.
        # NOTE prune_count_matrix is transposed (relative to confident_joint)
        prune_count_matrix = cleanlab.pruning.keep_at_least_n_per_class(
            prune_count_matrix=confident_joint.T,
            n=MIN_NUM_PER_CLASS,
        )

        s_counts = np.bincount(s)
        noise_masks_per_class = []
        # For each row in the transposed confident joint
        for k in range(K):
            noise_mask = np.zeros(len(psx), dtype=bool)
            psx_k = psx[:, k]
            if s_counts[k] > MIN_NUM_PER_CLASS:  # Don't prune if not MIN_NUM_PER_CLASS
                for j in range(K):  # noisy label index (k is the true label index)
                    if k != j:  # Only prune for noise rates, not diagonal entries
                        num2prune = prune_count_matrix[k][j]
                        if num2prune > 0:
                            # num2prune'th largest p(classk) - p(class j)
                            # for x with noisy label j
                            margin = psx_k - psx[:, j]
                            s_filter = s == j
                            threshold = -np.partition(
                                -margin[s_filter], num2prune - 1
                            )[num2prune - 1]
                            noise_mask = noise_mask | (s_filter & (margin >= threshold))
                noise_masks_per_class.append(noise_mask)
            else:
                noise_masks_per_class.append(np.zeros(len(s), dtype=bool))

        # Boolean label error mask
        label_errors_bool = np.stack(noise_masks_per_class).any(axis=0)

         # Remove label errors if given label == model prediction
        for i, pred_label in enumerate(psx.argmax(axis=1)):
            # np.all let's this work for multi_label and single label
            if label_errors_bool[i] and np.all(pred_label == s[i]):
                label_errors_bool[i] = False

        # Convert boolean mask to an ordered list of indices for label errors
        label_errors_idx = np.arange(len(s))[label_errors_bool]
        # self confidence is the holdout probability that an example
        # belongs to its given class label
        self_confidence = np.array(
            [np.mean(psx[i][s[i]]) for i in label_errors_idx]
        )
        margin = self_confidence - psx[label_errors_bool].max(axis=1)

        self.r = np.zeros(len(self.y))
        self.r[label_errors_idx] = -margin

        label_errors_idx = label_errors_idx[np.argsort(margin)]
        self.noise_set = label_errors_idx

    def get_noise_score(self):
        return self.r

    def get_noise_set(self, fracion=1.0):
        return self.noise_set
