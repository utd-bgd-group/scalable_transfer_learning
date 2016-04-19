import os
import numpy as np

swap = lambda x: (x[1], x[0])

def add_random_key(it):
    # make sure we get a proper random seed
    seed = int(os.urandom(4).encode('hex'), 16) 
    # create separate generator
    rs = np.random.RandomState(seed)
    # Could be randint if you prefer integers
    return ((rs.rand(), swap(x)) for x in it)

rdd_with_keys = (rdd
  # It will be used as final key. If you don't accept gaps 
  # use zipWithIndex but this should be cheaper 
  .zipWithUniqueId()
  .mapPartitions(add_random_key, preservesPartitioning=True))


# Next you can repartition, sort each partition and extract values:

n = rdd.getNumPartitions()

(rdd_with_keys
    # partition by random key to put data on random partition 
    .partitionBy(n)
    # Sort partition by random value to ensure random order on partition
    .mapPartitions(sorted, preservesPartitioning=True)
    # Extract (unique_id, value) pairs
    .values())
