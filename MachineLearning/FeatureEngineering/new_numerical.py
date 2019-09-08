#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from contextlib import contextmanager

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

sp = {'sep':'\n\n', 'end':'\n\n'}
url = 'https://assets.datacamp.com/production/repositories/4443/datasets/40af41a3b8739d0ac4b3f9f85ee43630ecbe7f0c/house_prices_train.csv'
df = pd.read_csv(url)


kf = KFold(n_splits=4, shuffle=True)

