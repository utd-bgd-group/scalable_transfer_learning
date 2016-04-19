from pyspark import SparkContext

sc = SparkContext(appName="PythonPartitioning")

rdd = sc.parallelize(range(100000)).zipWithIndex().map(lambda x:(int(x[1]), x[0])).cache()

# This will create 50 partitions of roughly the same size

rdd.partitionBy(50).saveAsTextFile('hdfs://cshadoop1/kxh132430/py_test/bag')

