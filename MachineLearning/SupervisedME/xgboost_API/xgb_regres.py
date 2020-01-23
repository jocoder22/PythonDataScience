import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import datasets

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")
        
cal_housing = datasets.fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

print2(X.head(), X.shape, y[:5])