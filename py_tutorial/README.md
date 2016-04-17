# Submit and run python script on spark

## Open your .bash_profile files, add the following lines:

```bash
export PYTHONHOME=/opt/python-2.7.11
export PYTHONPATH=/opt/python-2.7.11
export PATH=$PYTHONHOME/bin:$PATH
```

## Go to the current working directory of your wordcount.py files, type in:

```bash
spark-submit wordcount.py
```

## Or specify more parameters like:

```bash
spark-submit --master yarn-cluster --verbose --executor-memory 4G --executor-cores 7 --num-executors 6 wordcount.py
```

You can further configure spark, see [Submitting Applications](http://spark.apache.org/docs/latest/submitting-applications.html)
