import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split as spp

irisdata = datasets.load_iris()
features_ = irisdata.data[:, [2, 3]]
target_ = irisdata.target

print(np.unique(target_))

# split the dataset
X_train, X_test, y_train, y_test = spp(features_, target_, test_size=0.3,
                                       random_state=1, stratify=target_)

print('Labels counts in target:', np.bincount(target_))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))
print('Labels counts in y_test:', np.bincount(X_test))
print('Labels counts in y_test:', np.bincount(X_train))


y = [y_train, y_test]
for item in y:
    print(np.bincount(item))