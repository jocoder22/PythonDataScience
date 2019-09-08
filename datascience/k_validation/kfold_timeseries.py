#!/usr/bin/env python
import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import TimeSeriesSplit
from kfold_nonserial import changepath

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

path = 'C:\\Users\\Jose\\Desktop\\TimerSeriesAnalysis'
sp = {'sep':'\n\n', 'end':'\n\n'}


with changepath(path):
    df = pd.read_csv('AMZN.csv')

df['Date'] = pd.to_datetime(df['Date'])
df["year"] =  df.Date.dt.year

df.sort_values('Date')



print(df.head(), df.info(), **sp)


kf_timeseries_object = TimeSeriesSplit(n_splits=5)

k_fold = 0
# model = DecisionTreeRegressor(min_samples_leaf = 11 , min_samples_split = 33, random_state=500)

model_scores = []
model_rmse = []
for train_idx, test_idx in kf_timeseries_object.split(df):
    train_cv, test_cv = df.iloc[train_idx], df.iloc[test_idx]
    print(f'Fold: {k_fold}')
    print(f'Train fold shape: {train_cv.shape}')
    print(f'Train fold range: {train_cv.Date.min()} - {train_cv.Date.max()}')
    print(f'Test fold shape: {test_cv.shape}, Test fold range: {test_cv.Date.min()} - {test_cv.Date.max()}', **sp)
  
    k_fold += 1

    # model = DecisionTreeRegressor(min_samples_leaf = 11 , min_samples_split = 33, random_state=500)
    model = LinearRegression(normalize=True)
    model.fit(train_cv[['Open', 'Volume']], train_cv['Close'])
    
    pred = model.predict(test_cv[['Open', 'Volume']])
    score = model.score(test_cv[['Open', 'Volume']], test_cv['Close'])
    rmse = np.sqrt(mean_squared_error(test_cv['Close'], pred))
    model_scores.append(score)
    model_rmse.append(rmse)


mean_score = np.mean(model_scores) * 100
mean_rmse = np.mean(model_rmse)

overall_score = mean_score - np.std(model_scores) 
overall_rmse = mean_rmse + np.std(model_rmse)

print(f"{'Mean':>18} {'OverallScore':^21}")
print(f'Accuracy:    {mean_score:.02f}     {overall_score:.02f}')
print(f'RMSE    :    {mean_rmse:.02f}      {overall_rmse:.02f}')





