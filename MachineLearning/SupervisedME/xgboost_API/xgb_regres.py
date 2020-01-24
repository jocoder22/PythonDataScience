import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn import datasets

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")

# Load the California Housing Data     
calhousing = datasets.fetch_california_housing()
X = pd.DataFrame(calhousing.data, columns=calhousing.feature_names)
y = calhousing.target

print2(X.head(), X.shape, y[:5])


# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)


parameters = {'max_depth': 4, 'n_estimators':10, "booster":"gblinear", "objective":"reg:linear"}

# Instantiate the XGBClassifier: xgbRegress
xgbRegress = xgb.XGBRegressor(params=parameters)

# Convert the training and testing sets into DMatrixes: _train, _test
_train = xgb.DMatrix(data=X_train, label=y_train)
_test =  xgb.DMatrix(data=X_test, label=y_test)


# Train the model: xgbRegress
xgbRegress.fit(X_train,  y_train)

# Predict the labels of the test set: yhat
yhat = xgbRegress.predict(X_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test, yhat))
print("RMSE: %f" % (rmse))
