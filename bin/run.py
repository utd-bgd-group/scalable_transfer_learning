import numpy as np
from util import bagger
from util.splitter import split
# from util import kmm
# from pyspark import SparkContext


def main():
    train, train_beta, test = split('../dataset/powersupply.arff', 10000)
    train_idx = np.array(range(len(train)))
    test_idx = np.array(range(len(test)))
    sigma = 0.5
    print train[:9]
    print 'beta'
    print train_beta[:9]
    # tr_data = sc.broadcast(train)
    # te_data = sc.broadcast(test)

    bag_size = 1000
    tr_bag = bagger.bag(train_idx, size=bag_size)
    te_bag = bagger.bag(test_idx, size=bag_size)

    # bag_pairs = bagger.cartesian(tr_bag, te_bag)

    bag_pairs = bagger.random_combination(tr_bag, te_bag, 10)

    for bag in bag_pairs:
        print len(bag[0]), len(bag[1])

    # bag_index_rdd = sc.parallelize(bag_pairs)
    # bag_index_rdd.map(lambda (tr, te): kmm(tr_data.value[tr], te_data.value[te], sigma))
    # bag_index_rdd.collect()


if __name__ == '__main__':
    #sc = SparkContext(appName="PythonPartitioning", pyFiles=['lib.zip'])
    main()
