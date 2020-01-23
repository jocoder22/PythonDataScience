import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import datasets

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")

# Load the California Housing Data     
calhousing = datasets.fetch_california_housing()
X = pd.DataFrame(calhousing.data, columns=calhousing.feature_names)
y = cal_housing.target

print2(X.head(), X.shape, y[:5])


# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)


parameters = {'max_depth': 4, 'n_estimators':10, 'objective':'binary:logistic'}

# Instantiate the XGBClassifier: xgbooster
xgbooster = xgb.XGBClassifier(params=parameters)