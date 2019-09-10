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

