#!/usr/bin/env python
from pyspark import SparkContext
sc = SparkContext(appName="PythonProject", pyFiles=['lib.zip'])
import numpy as np
import sys
import argparse
from lib.splitter import split
from lib.bagger import bag, pair, cartesian
from lib.kmm import computeBeta
from lib.evaluation import computeNMSE


def get_bag_size(bag_size, data, sample_no):
    return bag_size if bag_size else len(data) / sample_no


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--bagging", type=int, choices=[1,2,3,4], default=1, help="bagging strategy")
    parser.add_argument("-t", "--training", type=int, default=12000, help="size of trainning data")
    parser.add_argument("-r", "--reverse", action="store_true", help="set -t as the size of test data")
    parser.add_argument("-s", "--bag_size", type=int, help="the sample size")
    parser.add_argument("-m", "--train_samples", type=int, default=1, help="number of samples from training")
    parser.add_argument("-n", "--test_samples", type=int, default=1, help="number of samples from test")
    parser.add_argument("-i", "--input", type=str, default='./dataset/powersupply.arff', help="default input file")
    parser.add_argument("-o", "--output", type=str, default='./nmse.txt', help="default output file")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    args = parser.parse_args()

    mode = args.bagging # bagging strategy
    training_size = args.training # training set size (small training set)
    reverse = args.reverse # flip training to test (small test set)
    bag_size = args.bag_size # By default, the bag size is dynamic, if specified, the bag size will fix
    m = args.train_samples # take m samples from training
    n = args.test_samples # take n samples from test
    input_file = args.input # input file path
    output_file = args.output # output file path

    # Generate biased train and test set, as well as the orginal beta for train set
    train, train_beta, test = split(input_file, training_size, reverse)
    train_data = np.array(train)
    test_data = np.array(test)
    orig_beta_data = np.array(train_beta)

    tr_bag_size = len(train_data)
    te_bag_size = len(test_data)
    # Generate the bagging index using different bagging strategies
    cartesian_sampling = True
    if mode == 1: # take m samples from train set and entire test set
        tr_bag_size = get_bag_size(bag_size, train_data, m)
        tr_n = bag(train_data, size=tr_bag_size, sample_no=m)
        te_n = [range(len(test_data))]
    elif mode == 2: # take entire train set and n samples from test set
        tr_n = [range(len(train_data))]
        te_bag_size = get_bag_size(bag_size, test_data, n)
        te_n = bag(test_data, size=te_bag_size, sample_no=n)
    elif mode == 3: # take m from train set and n samples from test set, do cartesian product pairing
        tr_bag_size = get_bag_size(bag_size, train_data, m)
        te_bag_size = get_bag_size(bag_size, test_data, n)
        tr_n = bag(train_data, size=tr_bag_size, sample_no=m)
        te_n = bag(test_data, size=te_bag_size, sample_no=n)
    elif mode == 4: # take min(m,n) samples each from train set and test set, do one-to-one pairing
        tr_bag_size = get_bag_size(bag_size, train_data, m)
        te_bag_size = get_bag_size(bag_size, test_data, n)
        tr_n = bag(train_data, size=tr_bag_size, sample_no=m)
        te_n = bag(test_data, size=te_bag_size, sample_no=n)
        cartesian_sampling = False
    else:
        print 'Invalid mode!'
        sys.exit(-1)

    # Bagging the train and test data from the sampled index
    if cartesian_sampling:
        bags = cartesian(train_data, test_data, tr_n, te_n)
    else:
        bags = pair(train_data, test_data, tr_n, te_n, sample_no=min(m, n))

    rdd = sc.parallelize(bags)

    # Compute the estimated beta
    res = rdd.map(lambda (idx, tr, te): computeBeta(idx, tr, te)).flatMap(lambda x: x)

    rdd1 = res.aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1),
                              lambda a,b: (a[0] + b[0], a[1] + b[1]))

    est_beta_map = rdd1.mapValues(lambda v: v[0]/v[1]).collectAsMap()
    est_beta_idx = est_beta_map.keys()

    # Compute the NMSE between the est_beta and orig_beta
    est_beta = [est_beta_map[x] for x in est_beta_idx]
    orig_beta = orig_beta_data[est_beta_idx]
    final_result = computeNMSE(est_beta, orig_beta)

    # statistics
    statistics = "mode=%s, train_size=%i, test_size=%i, bag_size=%s, m=%i, n=%i\n" % \
                 (mode, len(train_data), len(test_data), bag_size if bag_size else [tr_bag_size, te_bag_size], m, n)

    # Save the result into a text file
    with open(output_file, 'a') as output_file:
        message = "The final NMSE is : %s \n" % final_result
        print message
        output_file.write(statistics)
        output_file.write(message)

if __name__ == '__main__':
    main()


