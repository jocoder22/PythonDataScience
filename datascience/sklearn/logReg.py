#!/usr/bin/env python
import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split as splitit
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

irisdata = datasets.load_iris()
features_ = irisdata.data
target_ = irisdata.target

print(np.unique(target_))

# Standardization
stardard = StandardScaler()
# Standardization with the original data
stardard.fit(features_)
features_std = stardard.transform(features_)


# Split features into test and training datasets
X_train, X_test, y_train, y_test = splitit(features_std, target_, test_size=0.3,
                                       random_state=1, stratify=target_)

# fit the logistic regression
logreg = LogisticRegression(C=100.0, random_state=1)
logreg.fit(X_train, y_train)
print(logreg.predict_proba(X_test[:10, :]))
print(logreg.predict_proba(X_test[:10, :]).argmax(axis=1))
print(logreg.predict(X_test[:10, :]))

y_pred = logreg.predict(X_test)
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
print('Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))
# Accuracy: 97.78%