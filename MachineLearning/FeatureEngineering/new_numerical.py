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


def get_kfold_rmse(train):
    mse_scores = []

    for train_index, test_index in kf.split(train):
        train = train.fillna(0)
        feats = [x for x in train.columns if x not in ['Id', 'SalePrice', 'RoofStyle', 'CentralAir']]
        
        fold_train, fold_test = train.loc[train_index], train.loc[test_index]

        # Fit the data and make predictions
        # Create a Random Forest object
        rf = RandomForestRegressor(n_estimators=10, min_samples_split=10, random_state=123)

        # Train a model
        rf.fit(X=fold_train[feats], y=fold_train['SalePrice'])

        # Get predictions for the test set
        pred = rf.predict(fold_test[feats])
    
        fold_score = mean_squared_error(fold_test['SalePrice'], pred)
        mse_scores.append(np.sqrt(fold_score))
        
    return round(np.mean(mse_scores) + np.std(mse_scores), 2)


# Look at the initial RMSE
print('RMSE before feature engineering:', get_kfold_rmse(df))


# Add total area of the house
df2 = df.copy()
df2['TotalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
# df2.drop(columns=['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], inplace=True)
print('RMSE with total area:', get_kfold_rmse(df2))



# Add garden area of the property
df2['GardenArea'] = df['LotArea'] - df['1stFlrSF']
# df2.drop(columns=['LotArea'], inplace=True)
print('RMSE with garden area:', get_kfold_rmse(df2))


# Add total number of bathrooms
df2['TotalBath'] = df['FullBath']  + df['HalfBath']
# df2.drop(columns=['FullBath', 'HalfBath'], inplace=True)
print('RMSE with number of bathrooms:', get_kfold_rmse(df2))
