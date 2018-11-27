import numpy as np
from scipy import stats
from sklearn.datasets import load_boston

dataset = load_boston()
mydata, = dataset.data
mylabel, feature_names = dataset.target, dataset.feature_names

print(mydata.dtype)
print(mylabel.dtype)

np.isnan(np.sum(mydata))
np.isnan(np.sum(mylabel))

min = np.round(np.amin(mydata, axis=0), decimals=3)
max = np.round(np.amax(mydata, axis=0), decimals=3)
range = np.round(np.ptp(mydata, axis=0), decimals=3)
mean = np.round(np.mean(mydata, axis=0), decimals=3)
median = np.round(np.median(mydata, axis=0), decimals=3)
variance = np.round(np.var(mydata, axis=0), decimals=3)
percentile10 = np.round(np.percentile(mydata, 10, axis=0), decimals=3)
percentile90 = np.round(np.percentile(mydata, 90, axis=0), decimals=3)


mystats = np.vstack((min, max, range, mean, median,
                     variance, percentile10, percentile90))

statlabels = ["min", "max", "range", "mean",
              "median", "variance", "10%Tile", "90%Tile"]

statlabels2 = ["minn", "maxx", "rang", "mean",
               "medi", "vari", "10%T", "90%T"]

for labels, row in zip(statlabels2, mystats):
    print('{} [{}]'.format(labels, "".join('%07s' % i for i in row)))

for labels, row in zip(statlabels2, mystats):
    print(labels,  row)


# Using scipy
result = stats.describe(mydata, axis=0)

min2 = result.minmax[0]
max2 = result.minmax[1]
mean2 = result.mean
var = result.variance
percentile10 = stats.scoreatpercentile(mydata, per=10, axis=0)
percentile90 = stats.scoreatpercentile(mydata, per=90, axis=0)
range2 = stats.iqr(mydata, rng=(20, 80), axis=0)

