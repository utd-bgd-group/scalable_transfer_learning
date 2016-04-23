from sklearn.cross_validation import KFold
import numpy as np


def bag(data, size=50, random=True):
    n = len(data)
    # kf = KFold(n, n_folds=n / size, shuffle=random)
    # indexes = []
    # for remain, sample in kf:
    #     indexes.append(sample)
    indexes = []
    for i in range(n / size):
        sample = np.random.choice(n, size, False)
        indexes.append(sample)
    return indexes


def cartesian(train_index, test_index):
    bags = []
    for train in train_index:
        for test in test_index:
            bags.append((train, test))
    return bags


def random_combination(train_index, test_index, k=10):
    bags = []
    train = np.random.choice(len(train_index), k, replace=False)
    test = np.random.choice(len(test_index), k, replace=False)
    for i in range(k):
        bags.append((train_index[train[i]], test_index[test[i]]))
    return bags

