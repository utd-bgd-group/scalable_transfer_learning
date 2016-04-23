from scipy.io import arff
from sklearn.cross_validation import train_test_split
from examples.kmm import kmm
from examples.data import *

data, meta = arff.loadarff('../dataset/powersupply.arff')
orig_data = []
for line in data:
    orig_data.append(list(line)[0:-1])

train, trainBeta, data = generateTrain(tuple(orig_data), trainsize=2)

print 'result:'
print train
print trainBeta
print data[0]


# X_train, X_test = train_test_split(orig_data, test_size=0.4, random_state=0)
#
# # Here I use a arbitrary sigma
# beta, running_time = kmm(X_train, X_test, sigma = 0.5)
# print "Beta vector p(x_tr)/p(x_ts)= %s " % beta
# print 'Total running time is: %s ms' % running_time
