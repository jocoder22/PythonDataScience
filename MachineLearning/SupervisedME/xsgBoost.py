import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import datasets

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")
        
        
# Load the load_breast_cancer dataset
breast_X, breast_y = datasets.load_breast_cancer(return_X_y=True)

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(
    breast_X, breast_y, test_size=0.2, random_state=42)


parameters = {'max_depth': 4, 'n_estimators':10, 'objective':'binary:logistic'}

# Instantiate the XGBClassifier: xgbooster
xgbooster = xgb.XGBClassifier(params=parameters)

# Fit the classifier to the training set
xgbooster.fit(X_train, y_train)

# Predict the labels of the test set: yhat
yhat = xgbooster.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(yhat==y_test))/y_test.shape[0] *100
print2(f"Accuracy: {accuracy:.2f}%")


# Create the DMatrix from X and y: xgbdmatrix
xgbdmatrix = xgb.DMatrix(data=breast_X, label=breast_y)

# Update the parameter dictionary
parameters["objective"] = "reg:logistic"

# Perform cross-validation: results
results = xgb.cv(dtrain=xgbdmatrix, params=parameters, 
                    nfold=3, num_boost_round=5, 
                    metrics=["error", "auc"], as_pandas=True)

# Print results
print2(results)

# Print the accuracy, Area Under Receiver Operating Characteristic Curve
print2(f'Accuracy : {1-results["test-error-mean"].iloc[-1]:.2f}')
print2(f'AUC : {results["test-auc-mean"].iloc[-1]:.2f}')