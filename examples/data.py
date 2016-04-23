import math, numpy, sklearn.metrics.pairwise as sk
import logging, random

# create logger with 'spam_application'
logger = logging.getLogger('spam_application')

# data can be an iterable of (map, list, ...) sparse map

#Compute mean in sparse format


def computeMean(data):
    mean = {}
    for d in data:
        for (k, v) in enumerate(d):
            if mean.has_key(k):
                mean[k] += v #d[i]
            else:
                mean[k] = v #d[i]
    print mean
    for (k, v) in mean.iteritems():
        mean[k] = v / len(data)
    return mean


#Compute distance
def computeDistanceSq(d1, d2):
    dist = 0
    for i, e in enumerate(d1):
            dist += (e - d2[i]) ** 2
    return dist

#Compute standard deviation from mean of sparse data
def computeSTD(data, mean):
    print 'computing distance for STD'
    distList = []
    for d in data:
        dist = computeDistanceSq(d, mean)
        distList.append(dist)

    print 'Computing STD'
    return numpy.std(distList)


#Compute probability of sampling instance as training data
def computeProb(instance, mean, std):
    dist = computeDistanceSq(instance, mean)
    return math.exp(-1 * dist / std)


#Sample training data from input sparse data
def generateTrain(origdata, trainsize):
    train = []
    trainBeta = []
    data = list(origdata)
    print 'The total size is: %i' % len(data)

    logger.info('Computing mean & std')
    mean = computeMean(data)
    std = computeSTD(data, mean)

    logger.info('Generating training data')

    count = 0
    stdcount = 1
    origstd = std

    while len(train) < trainsize:
        d = random.randint(0, len(data)-1)
        p = computeProb(data[d], mean, std)
        x = random.uniform(0,1)
        if x < p:
            train.append(data[d])
            trainBeta.append(1.0/p)
            del data[d]
            count = 0
        else:
            count += 1

        if count == trainsize:
            count = 0
            stdcount += 1
            std = origstd*stdcount

    return train, trainBeta, data
