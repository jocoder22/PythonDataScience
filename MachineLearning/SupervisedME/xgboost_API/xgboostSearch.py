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

print2(y[:10], X.head())

"""
_grid = {
    'colsample_bytree': np.arange(0.2, 0.8, 0.1),
    'n_estimators': np.arange(20, 100, 10),
    'max_depth': np.arange(2,5,1)
}

# Instantiate the XGBClassifier: xgbReg
xgbReg = xgb.XGBRegressor(objective='reg:squarederror')


#########################################################
#########    Using Grid search    #######################
#########################################################    
# Perform grid search: 
tnow = time.time()
gridsearcher = GridSearchCV(estimator=xgbReg, param_grid=_grid,
                        scoring='neg_mean_squared_error', cv=4, verbose=1, n_jobs=-1)
gridsearcher.fit(X, y)
tlater = time.time()
print(f'Elapsed time: {round(tlater - tnow, 3)} seconds')

# Print the best parameters and lowest RMSE
print2(f"Best parameters found: {gridsearcher.best_params_}")
print2(f"Lowest RMSE found: {np.sqrt(np.abs(gridsearcher.best_score_)):.3f}")


###########################################################
#########    Using Random search    ####################### 
###########################################################
# Perform random search: 
tnow = time.time()
randomsearcher = RandomizedSearchCV(estimator=xgbReg, param_distributions=_grid, n_iter=100,
                        scoring='neg_mean_squared_error', cv=4, verbose=1, n_jobs=-1)
randomsearcher.fit(X, y)
tlater = time.time()
print(f'Elapsed time: {round(tlater - tnow, 3)} seconds')

# Print the best parameters and lowest RMSE
print2(f"Best parameters found: {randomsearcher.best_params_}")
print2(f"Lowest RMSE found: {np.sqrt(np.abs(randomsearcher.best_score_)):.3f}")

"""
how_many_snakes = 14
snake_string = """
Welcome to Python3!

             ____
            / . .\\   Hello Python
            \  ---<
             \  /     Ninga!!!
   __________/ /
-=:___________/

<3, Juno
"""


print(snake_string * how_many_snakes)

