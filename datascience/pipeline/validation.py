from sklearn.datasets import load_boston, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, 
                                                test_size=0.50, random_state=4)

iclassifier = DecisionTreeClassifier(max_depth=2)
iclassifier.fit(xtrain, ytrain)
ypred = iclassifier.predict(xtest)
iris.target_names

# Performance metrics
c_metric = confusion_matrix(ytest, ypred)
print(c_metric)


# Classified wine dataset
wine = load_wine()
wtrain, wtest, wytrain, wytest = train_test_split(wine.data, wine.target,
                                                  test_size=0.50,
                                                  random_state=4)

ibclassifier = DecisionTreeClassifier(max_depth=2)
ibclassifier.fit(wtrain, wytrain)
wpred = ibclassifier.predict(wtest)
wine.target_names
list(wine.target_names)
wine.target[[20, 65, 123, 171]]

cb_metric = confusion_matrix(wytest, wpred)
print(cb_metric)




