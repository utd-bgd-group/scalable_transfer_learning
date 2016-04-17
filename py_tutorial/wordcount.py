from __future__ import print_function
from operator import add
from pyspark import SparkContext

sc = SparkContext(appName="PythonWordCount")

lines = sc.textFile('hdfs://cshadoop1/kxh132430/py_test/in/dict.txt', 1)

counts = lines.flatMap(lambda x: x.split(' ')).map(lambda x: (x, 1)).reduceByKey(add)

output = counts.collect()

for (word, count) in output:
    print("%s: %i" % (word, count))

counts.saveAsTextFile('hdfs://cshadoop1/kxh132430/py_test/out')

sc.stop()
