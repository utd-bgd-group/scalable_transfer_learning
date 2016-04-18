from __future__ import print_function
from pyspark import SparkContext
import numpy as np

sc = SparkContext(appName="PythonParallelProcessing")

def get_sum(iterator):
    a = []
    for x in iterator:
        a.append(x)
    # print (a)
    return [np.sum(a)]

def main():
    rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
    # rdd.foreachPartition(do something)
    res = rdd.mapPartitions(get_sum, True).collect()
  
    resRdd = sc.parallelize(res)
    resRdd.saveAsTextFile('hdfs://cshadoop1/kxh132430/py_test/out_1')

if __name__ == '__main__':
	main()
