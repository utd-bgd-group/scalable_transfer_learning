import numpy as np
import math


def bag(data, size, sample_no):
    n = len(data)
    data_n = []
    for i in range(min(sample_no, n / size)):
        index = np.random.choice(n, size, False)
        data_n.append(index)
    return data_n


def cartesian(train, test, tr_n, te_n):
    bags = []
    for tr in tr_n:
        for te in te_n:
            bags.append((tr, train[tr], test[te]))
    return bags


def pair(train, test, tr_n, te_n, k=10):
    bags = []
    tr_bag_indices = np.random.choice(len(tr_n), k, replace=False)
    te_bag_indices = np.random.choice(len(te_n), k, replace=False)
    for i in range(k):
        tr_idx = tr_n[tr_bag_indices[i]]
        te_idx = te_n[te_bag_indices[i]]
        bags.append((tr_idx, train[tr_idx], test[te_idx]))
    return bags

