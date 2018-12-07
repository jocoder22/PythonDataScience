from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, 
                                                test_size=0.50, random_state=4)

iclassifier = DecisionTreeClassifier(max_depth=2)
iclassifier.fit(xtrain, ytrain)
ypred = iclassifier.predict(xtest)
iris.target_names