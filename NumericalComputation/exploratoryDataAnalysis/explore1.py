import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
mydata, mylabel, feature_name = dataset.data, dataset.target, dataset.feature_names

print(mydata.dtype)
print(mylabel.dtype)

np.isnan(np.sum(mydata))
np.isnan(np.sum(mylabel))
