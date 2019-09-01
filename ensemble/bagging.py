#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

plt.style.use('ggplot')
sp ={'sep':'\n\n', 'end':'\n\n'}

path = r'C:\Users\Jose\Desktop\PythonDataScience\ensemble'

os.chdir(path)

npzfile = np.load('mydata.npz')
X_train, X_test =  npzfile['X_train'], npzfile['X_test']
y_train, y_test =npzfile['y_train'], npzfile['y_test']