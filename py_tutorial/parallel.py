import numpy as np

def f(iterator):
    a = []
    for x in iterator:
        a.append(x)
    print a
    print 'sum= %i ' % np.sum(a)
    yield 10

sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3).foreachPartition(f)