#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import BaggingClassifier
# from contextlib import contextmanager

import sys
path = r'C:\Users\Jose\Desktop\PythonDataScience\ensemble'

sys.path.insert(0, path)

from ctmanager import functionX as mgt

plt.style.use('ggplot')
sp = {'sep':'\n\n', 'end':'\n\n'}

print(dir(mgt), **sp)

with mgt.changepath(path):
    npzfile = np.load('mydata_l.npz')
    X_train, X_test =  npzfile['X_train'], npzfile['X_test']
    y_train, y_test =npzfile['y_train'], npzfile['y_test']


# Linear model to predict life expectancy
# Initialize LinearRegression model
lreg_model = LinearRegression(normalize=True)

# Fit the model
lreg_model.fit(X_train, y_train)

# Predict and score the model
pred = lreg_model.predict(X_test)
score = lreg_model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))

# print the rmse and accuracy score
print(f'RMSE: {rmse:.02f}\nAccuracy: {score:.02f}%', **sp)



lm_adaB = AdaBoostRegressor(lreg_model, n_estimators=12, random_state=500)
lm_adaB.fit(X_train, y_train)

# Calculate the predictions on the test set
pred_adaB = lm_adaB.predict(X_test)

# Evaluate the performance using the RMSE
score_adaB = lm_adaB.score(X_test, y_test)
rmsea = np.sqrt(mean_squared_error(y_test, pred_adaB))
print(f'RMSE: {rmsea:.02f}\nAccuracy: {score_adaB:.02f}%', **sp)