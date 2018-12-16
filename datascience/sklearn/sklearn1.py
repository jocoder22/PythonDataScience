import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as split

irisdata = datasets.load_iris()
features_ = irisdata.data[:, [2, 3]]
target_ = irisdata.target

print(np.unique(target_))

# split the dataset
X_train, X_test, y_train, y_test = split(features_, target_, test_size=0.3,
                                       random_state=1, stratify=target_)

print('Labels counts in target:', np.bincount(target_))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Standardization
stardard = StandardScaler()
stardard.fit(X_train)
X_train_std = stardard.transform(X_train)
X_test_std = stardard.transform(X_test)