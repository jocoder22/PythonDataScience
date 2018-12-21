#!/usr/bin/env python

import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split as splitit
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


iris_ = datasets.load_iris()
iris_features = iris_.data
iris_target = iris_.target
print(np.unique(iris_target))
print(iris_features[:5, :])


digits_ = datasets.load_digits()
digits_featrues = digits_.data
digits_target = digits_.target
print(np.unique(digits_target))
print(digits_featrues[:5, :])

# split the dataset for training and testing
Xi_train, Xi_test, yi_train, yi_test = splitit(iris_features, iris_target, test_size=0.3,
                                       random_state=1, stratify=iris_target)

Xd_train, Xd_test, yd_train, yd_test = splitit(digits_featrues, digits_target, test_size=0.3,
                                       random_state=1, stratify=digits_target)

# fit the Decision tree model
mytree1 = DecisionTreeClassifier(criterion='gini', max_depth=4, 
                            random_state=1)

mytree1.fit(Xi_train, yi_train)
yi_pred = mytree1.predict(Xi_test)
print('Accuracy: {:.3f}'.format(accuracy_score(yi_test, yi_pred)))
print('Accuracy: {:.2f}%'.format(accuracy_score(yi_test, yi_pred) * 100))
# Accuracy: 97.78%



