import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
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


parameters = {'max_depth': 4, 'n_estimators':10, 'booster':'gblinear'}

# Instantiate the XGBClassifier: xgbRegress
xgbRegress = xgb.XGBRegressor(objective='reg:squarederror', params=parameters)

# Convert the training and testing sets into DMatrixes: _train, _test
_train = xgb.DMatrix(data=X_train, label=y_train)
_test =  xgb.DMatrix(data=X_test, label=y_test)


# Train the model: xgbRegress
xgbRegress.fit(X_train,  y_train)

# Predict the labels of the test set: yhat
yhat = xgbRegress.predict(X_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test, yhat))
mae = mean_absolute_error(y_test, yhat)
print2(f"RMSE: {rmse}", f"MAE Score: {mae}")



# Create the DMatrix: California Housing Data 
_dmatrix = xgb.DMatrix(data=X, label=y)

parameters = {'objective':'reg:squarederror', 'max_depth': 4}

# Perform cross-validation: _results
_results = xgb.cv(dtrain=_dmatrix,  params=parameters, nfold=4,
                  num_boost_round=8, metrics=["error", "mae", "rmse"], as_pandas=True)

# Print _results
print(_results)

# Extract and print final boosting round metric
print((_results["test-rmse-mean"]).tail(1))


# Create the DMatrix: reg_params
regparams = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective":"reg:squarederror", "max_depth":3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses = []

# Iterate over reg_params
for reg in regparams:

    # Update l2 strength
    params["lambda"] = reg
    
    # Pass this updated param dictionary into cv
    _results_rmse = xgb.cv(dtrain=_dmatrix, params=params, nfold=4, 
                             num_boost_round=5, metrics="rmse", as_pandas=True)
    
    # Append best rmse (final round) to rmses_l2
    rmses.append(_results_rmse["test-rmse-mean"].tail(1).values[0])


# Look at best rmse 
df = pd.DataFrame(list(zip(regparams, rmses)), columns=["l2","rmse"])
print2(df)

# Plotting the trees for visualization
# Train the model: _reg
_reg = xgb.train(params=params, dtrain=_dmatrix, num_boost_round=12)

# Plot the first tree
xgb.plot_tree(_reg, num_trees=0)
plt.show()

# Plot the second tree
xgb.plot_tree(_reg, num_trees=1)
plt.show()

# Plot the fifth tree
xgb.plot_tree(_reg, num_trees=4)
plt.show()

# Plot the ninth tree
xgb.plot_tree(_reg, num_trees=8)
plt.show()



# Plot the last tree sideways
xgb.plot_tree(_reg, num_trees=9, rankdir="LR")
plt.show()


# Plot the feature importances
xgb.plot_importance(_reg)
plt.show()
