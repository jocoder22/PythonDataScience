#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from contextlib import contextmanager

@contextmanager
def changepath(path):
    currentpath = os.getcwd()
    os.chdir(path)

    try:
        yield 

    finally:
        os.chdir(currentpath)

plt.style.use('ggplot')
sp = {'sep':'\n\n', 'end':'\n\n'}

path = r'C:\Users\Jose\Desktop\PythonDataScience\ensemble'


with changepath(path):
    npzfile = np.load('mydata.npz')
    X_train, X_test =  npzfile['X_train'], npzfile['X_test']
    y_train, y_test =npzfile['y_train'], npzfile['y_test']


 # Instantiate the base model
 # The base model is using a weak model, here we limit the max_depth
#  for linear model, we use normalized=False argument
_dt = DecisionTreeClassifier(max_depth=4)

# Build and train the Bagging classifier
_bag = BaggingClassifier(
  base_estimator=_dt,
  n_estimators=21,
  random_state=500)
 
 
_bag.fit(X_train, y_train)

# Predict the labels of the test set
pred = _bag.predict(X_test)

# Show the F1-score
print('F1-Score: {:.2f}'.format(accuracy_score(y_test, pred)))
print(f'Score for the BaggingClasifier: {_bag.score(X_test, y_test):.2f}')