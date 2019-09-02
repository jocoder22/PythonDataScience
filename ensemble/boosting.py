#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.linear_model import LogisticRegression
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
    npzfile = np.load('mydata.npz')
    X_train, X_test =  npzfile['X_train'], npzfile['X_test']
    y_train, y_test =npzfile['y_train'], npzfile['y_test']

