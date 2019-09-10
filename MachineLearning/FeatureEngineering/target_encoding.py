#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from contextlib import contextmanager

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

sp = {'sep':'\n\n', 'end':'\n\n'}
path = r'C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\FeatureEngineering'

os.chdir(path)
df = pd.read_csv('housing.csv')

kf = KFold(n_splits=4, shuffle=True, random_state=23)



def test_encoding(train, test, target, cat, alpha=7):
    # global mean on the train data
    mean_global = train[target].mean()
    
    # Get categorical feature sum and size
    cat_sum = train.groupby(cat)[target].sum()
    cat_size = train.groupby(cat).size()
    
    # smoothed  statistics
    train_smoothed = (cat_sum + mean_global * alpha) / (cat_size + alpha)
    
    # get encodings for  test data
    test_encoded = test[cat].map(train_smoothed).fillna(mean_global)
    return test_encoded.values

