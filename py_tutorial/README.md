# Submit and run python script on spark

## Open your .bash_profile files, add the following lines:

```bash
export PYTHONHOME=/opt/python-2.7.11
export PYTHONPATH=/opt/python-2.7.11
export PATH=$PYTHONHOME/bin:$PATH
```

## Go to the current working directory of your wordcount.py files, type
in the following commands in the console:

```bash
spark-submit wordcount.py
```
You can further configure spark, see [Submitting Applications](http://spark.apache.org/docs/latest/submitting-applications.html)
