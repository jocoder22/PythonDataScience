import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import datasets

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")

# Load the California Housing Data     
calhousing = datasets.fetch_california_housing()
X = pd.DataFrame(calhousing.data, columns=calhousing.feature_names)
y = calhousing.target


print2(X[:6], y[:6], X.dtypes)