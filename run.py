#!/usr/bin/env python
from pyspark import SparkContext
sc = SparkContext(appName="PythonProject", pyFiles=['lib.zip'])
import numpy as np
import argparse
import time
from lib.splitter import split
from lib.bagger import bag, pair, cartesian
from lib.kmm import computeBeta
from lib.evaluation import computeNMSE


def get_size_no(data, bag_size, sample_no):
    if bag_size:
        if sample_no:
            return bag_size, sample_no
        else:
            return bag_size, len(data) / bag_size
    else:
        if sample_no:
            return len(data) / sample_no, sample_no
        else:
            return len(data), 1


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

    # Step 1: Generate biased train and test set, as well as the orginal beta for train set
    start = time.time()

    train, train_beta, test = split(input_file, training_size, reverse)
    train_data = np.array(train)
    test_data = np.array(test)
    orig_beta_data = np.array(train_beta)

    end = time.time()
    split_time = end - start

    # Step 2: Generate the bagging index using different bagging strategies
    start = time.time()

    # Bagging the train and test data from the sampled index
    tr_bag_size, tr_bag_no = get_size_no(train_data, bag_size, m)
    te_bag_size, te_bag_no = get_size_no(test_data, bag_size, n)
    tr_n = bag(train_data, size=tr_bag_size, sample_no=tr_bag_no)
    te_n = bag(test_data, size=te_bag_size, sample_no=te_bag_no)

    if mode < 4:
        bags = cartesian(train_data, test_data, tr_n, te_n)
    else:
        bags = pair(train_data, test_data, tr_n, te_n, sample_no=min(tr_bag_no, te_bag_no))
    rdd = sc.parallelize(bags)

    end = time.time()
    bagging_time = end - start

    # Step 3: Compute the estimated beta
    start = time.time()

    res = rdd.map(lambda (idx, tr, te): computeBeta(idx, tr, te)).flatMap(lambda x: x)

    rdd1 = res.aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1),
                              lambda a,b: (a[0] + b[0], a[1] + b[1]))

    est_beta_map = rdd1.mapValues(lambda v: v[0]/v[1]).collectAsMap()
    est_beta_idx = est_beta_map.keys()

    end = time.time()
    compute_time = end - start

    # Step 4: Compute the NMSE between the est_beta and orig_beta
    start = time.time()

    est_beta = [est_beta_map[x] for x in est_beta_idx]
    orig_beta = orig_beta_data[est_beta_idx]
    final_result = computeNMSE(est_beta, orig_beta)

    end = time.time()
    evaluate_time = end - start

    # statistics
    statistics = "mode=%s, train_size=%i, test_size=%i, tr_bag_size=%i, m=%i, te_bag_size=%i, n=%i\n" % \
                 (mode, len(train_data), len(test_data), tr_bag_size, tr_bag_no, te_bag_size, te_bag_no)
    total_time = split_time + bagging_time + compute_time + evaluate_time
    time_info = "split_time=%s, bagging_time=%s, compute_time=%s, evaluate_time=%s, total_time=%s\n" % \
                (split_time, bagging_time, compute_time, evaluate_time, total_time)
    print statistics
    print time_info

    # Save the result into a text file
    with open(output_file, 'a') as output_file:
        message = "The final NMSE is : %s \n" % final_result
        print message
        output_file.write(statistics)
        output_file.write(time_info)
        output_file.write(message)

if __name__ == '__main__':
    main()


