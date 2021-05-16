#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from mlxtend.classifier import StackingClassifier
from mlxtend.regressor import StackingRegressor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
# from sklearn.svm import SVC
# from sklearn.ensemble import BaggingClassifier
# from contextlib import contextmanager

import sys
path = r'C:\Users\Jose\Desktop\PythonDataScience\ensemble'

sys.path.insert(0, path)

from ctmanager import functionX as mgt

plt.style.use('ggplot')
sp = {'sep':'\n\n', 'end':'\n\n'}

print(dir(mgt), **sp)



with mgt.changepath(path):
    npzfile = np.load('mydata_l.npz')
    X_train, X_test =  npzfile['X_train'], npzfile['X_test']
    y_train, y_test =npzfile['y_train'], npzfile['y_test']


# Linear model to predict life expectancy
# Initialize LinearRegression model
lreg_model = LinearRegression(normalize=True)

# Fit the model
lreg_model.fit(X_train, y_train)

# Predict and score the model
pred = lreg_model.predict(X_test)
score = lreg_model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))

# print the rmse and accuracy score
print(f'''RMSE simple Linear Regression model: {rmse:.02f}
Accuracy simple Linear Regression model: {score:.02f}%''', **sp)



#############################################################################
# Add AdaBoost model
# Initialize the model with linearRegression parameters
lm_adaB = AdaBoostRegressor(lreg_model, n_estimators=12, random_state=500)
lm_adaB.fit(X_train, y_train)

# Predict using the AdaBoost model
pred_adaB = lm_adaB.predict(X_test)

# Evaluate the performance using the RMSE and Accuracy score
score_adaB = lm_adaB.score(X_test, y_test)
rmsea = np.sqrt(mean_squared_error(y_test, pred_adaB))
print(f'''
        RMSE AdaBoost Linear Regression model: {rmsea:.02f}
        Accuracy AdaBoost Linear Regression model: {score_adaB:.02f}%
        ''', **sp)

# Add AdaBoost model
# Initialize the model with default Tree based regression model
lm_adaBT = AdaBoostRegressor( n_estimators=250, random_state=500, learning_rate=0.01)
lm_adaBT.fit(X_train, y_train)


# Predict using the AdaBoost model
pred_adaBT = lm_adaBT.predict(X_test)


# Evaluate the performance using the RMSE and Accuracy score
score_adaBT = lm_adaBT.score(X_test, y_test)
rmseaT = np.sqrt(mean_squared_error(y_test, pred_adaBT))
print(f'RMSE Tree based(Defualt): {rmseaT:.02f}\nAccuracy Tree based(Defualt): {score_adaBT:.02f}%', **sp)


# Build and fit a CatBoost regressor
_cat = CatBoostRegressor(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=500)
_cat.fit(X_train, y_train)

# Calculate the predictions on the set set
pred = _cat.predict(X_test)



# Evaluate the performance using the RMSE
s_cat = _cat.score(X_test, y_test)
rmse_cat = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE (CatBoost): {:.3f}, Accuracy Score: {:.2f}'.format(rmse_cat, s_cat), **sp)



# Build and fit a XGBoost regressor
_xgb = xgb.XGBRegressor(objective='reg:squarederror', max_depth = 3, learning_rate = 0.1,  n_estimators = 100, random_state=500)
_xgb.fit(X_train, y_train)

params = {'objective':'reg:squarederror', 
          'max_depth' : 3, 
          'learning_rate' : 0.1,  
          'n_estimators' : 100, 
          'random_state':500}

dtrain = xgb.DMatrix(data=X_train,
                     label=y_train)

_xg_depth_3 = xgb.train(params=params, dtrain=dtrain)

# Build and fit a LightGBM regressor
_lgb = lgb.LGBMRegressor(max_depth = 3, learning_rate = 0.1,  n_estimators = 100, seed=500)
_lgb.fit(X_train, y_train)

# Calculate the predictions and evaluate both regressors
pred_xgb = _xgb.predict(X_test)
sc = _xgb.score(X_test, y_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
pred_lgb = _lgb.predict(X_test)
sc2 = _lgb.score(X_test, y_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))

print('''Extreme::: RMSE: {:.3f}, Accuracy: {:.2f}
Light::: RMSE: {:.3f}, Accuracy: {:.2f}'''.format(rmse_xgb, sc, rmse_lgb, sc2), **sp)





# Instantiate the 1st-layer regressors
reg_dt = DecisionTreeRegressor(min_samples_leaf = 11 , min_samples_split = 33, random_state=500)
reg_lr = LinearRegression(normalize=True)
reg_ridge = Ridge(random_state=500)

# Instantiate the 2nd-layer regressor
reg_meta = LinearRegression()

# Build the Stacking regressor
reg_stack = StackingRegressor(
    regressors=[reg_dt, reg_lr, reg_ridge],
    meta_regressor=reg_meta)
reg_stack.fit(X_train, y_train)

# Evaluate the performance on the test set using the MAE metric
pred = reg_stack.predict(X_test)
stacks = reg_stack.score(X_test, y_test)
rmseS = np.sqrt(mean_squared_error(y_test, pred))
print('MAE: {:.3f}'.format(mean_absolute_error(y_test, pred)))
print('RMSE (Stacking): {:.3f}, Accuracy Score: {:.2f}'.format(rmseS, stacks), **sp)





with mgt.changepath(path):
    npzfile = np.load('mydata.npz')
    X_train, X_test =  npzfile['X_train'], npzfile['X_test']
    y_train, y_test =npzfile['y_train'], npzfile['y_test']


# Create the first-layer models
clf_dt = DecisionTreeClassifier(min_samples_leaf = 5 , min_samples_split = 15, random_state=500)
clf_knn = KNeighborsClassifier(n_neighbors=12,algorithm='ball_tree' )
clf_nb = GaussianNB()
clf_lg = lgb.LGBMClassifier(max_depth = 3, learning_rate = 0.1,  n_estimators = 100, seed=500)
clf_ex = xgb.XGBClassifier(max_depth = 3, learning_rate = 0.1,  n_estimators = 100, random_state=500)
# Create the second-layer model (meta-model)
clf_lr = LogisticRegression(multi_class='multinomial', solver = 'lbfgs')

# Create and fit the stacked model
clf_stack2 = StackingClassifier(
    classifiers=[ clf_lr, clf_dt, clf_ex, clf_lg, clf_nb],
    meta_classifier=clf_knn)
clf_stack2.fit(X_train, y_train)

# Evaluate the stacked model’s performance
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, clf_stack2.predict(X_test))))