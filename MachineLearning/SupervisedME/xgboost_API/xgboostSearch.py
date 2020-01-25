import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import datasets

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")

# Load the California Housing Data     
calhousing = datasets.fetch_california_housing()
X = pd.DataFrame(calhousing.data, columns=calhousing.feature_names)
y = calhousing.target



_grid = {
    'colsample_bytree': np.arange(0.2, 0.8, 0.1),
    'n_estimators': np.arange(20, 100, 10),
    'max_depth': np.arange(2,6,1)
}

# Instantiate the XGBClassifier: xgbReg
xgbReg = xgb.XGBRegressor(objective='reg:squarederror')


#########    Using Grid search    #######################    
# Perform grid search: 
tnow = time.time()
gridsearcher = GridSearchCV(estimator=xgbReg, param_grid=_grid,
                        scoring='neg_mean_squared_error', cv=4, verbose=1, n_jobs=-1)
gridsearcher.fit(X, y)
tlater = time.time()
print(f'Elapsed time: {round(tlater - tnow, 3)} seconds')

# Print the best parameters and lowest RMSE
print("Best parameters found: ", gridsearcher.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(gridsearcher.best_score_)))



#########    Using Random search    ####################### 