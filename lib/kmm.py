import math, numpy, sklearn.metrics.pairwise as sk
import time
from cvxopt import matrix, solvers
from distance import compute_sigma


def kmm(Xtrain, Xtest, sigma):
    n_tr = len(Xtrain)
    n_te = len(Xtest)

    #calculate Kernel
    print 'Computing kernel for training data ...'
    K_ns = sk.rbf_kernel(Xtrain, Xtrain, sigma)
    #make it symmetric
    K = 0.5*(K_ns + K_ns.transpose())

    #calculate kappa
    print 'Computing kernel for kappa ...'
    kappa_r = sk.rbf_kernel(Xtrain, Xtest, sigma)
    ones = numpy.ones(shape=(n_te, 1))
    kappa = numpy.dot(kappa_r, ones)
    kappa = -(float(n_tr)/float(n_te)) * kappa

    #calculate eps
    eps = (math.sqrt(n_tr) - 1)/math.sqrt(n_tr)

    #constraints
    A0 = numpy.ones(shape=(1,n_tr))
    A1 = -numpy.ones(shape=(1,n_tr))
    A = numpy.vstack([A0, A1, -numpy.eye(n_tr), numpy.eye(n_tr)])
    b = numpy.array([[n_tr*(eps+1), n_tr*(eps-1)]])
    b = numpy.vstack([b.T, -numpy.zeros(shape=(n_tr,1)), numpy.ones(shape=(n_tr,1))*1000])

    print 'Solving quadratic program for beta ...'
    P = matrix(K, tc='d')
    q = matrix(kappa, tc='d')
    G = matrix(A, tc='d')
    h = matrix(b, tc='d')

    start = time.time()
    beta = solvers.qp(P, q, G, h)
    end = time.time()
    return [i for i in beta['x']], (end - start)


def computeBeta(idx, train, test):
    sigma = compute_sigma(train)
    res = kmm(train, test, sigma)
    beta = res[0]
    return [(x, y) for x, y in zip(idx, beta)]
