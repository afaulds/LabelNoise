from scipy.stats import ttest_ind
import numpy as np


class StatCompare:

    def mean(stat_a):
        stat = {}
        if len(stat_a) == 0:
            return {}
        keys = list(stat_a[0].keys())
        for key in keys:
            a = []
            for i in range(len(stat_a)):
                a.append(stat_a[i][key])
            stat[key] = np.mean(a)
        return stat

    def std(stat_a):
        stat = {}
        if len(stat_a) == 0:
            return {}
        keys = list(stat_a[0].keys())
        for key in keys:
            a = []
            for i in range(len(stat_a)):
                a.append(stat_a[i][key])
            stat[key] = np.std(a)
        return stat

    def diff(stat_a, stat_b):
        stat = {}
        keys = list(stat_a[0].keys())
        for key in keys:
            a = []
            for i in range(len(stat_a)):
                a.append(stat_a[i][key])
            b = []
            for i in range(len(stat_b)):
                b.append(stat_b[i][key])
            zscore, prob = ttest_ind(a, b, equal_var=False)
            stat[key] = {
                "p-value": prob,
                "a-mean": np.mean(a),
                "b-mean": np.mean(b)
            }
        return stat
