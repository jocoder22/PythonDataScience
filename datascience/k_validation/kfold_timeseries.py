#!/usr/bin/env python
import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import TimeSeriesSplit
from kfold_nonserial import changepath

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
for train_idx, test_idx in kf_timeseries_object.split(df):
    train_cv, test_cv = df.iloc[train_idx], df.iloc[test_idx]
    print(f'Fold: {k_fold}')
    print(f'Train fold shape: {train_cv.shape}')
    print(f'Train fold range: {train_cv.Date.min()} - {train_cv.Date.max()}')
    print(f'Test fold shape: {test_cv.shape}, Test fold range: {test_cv.Date.min()} - {test_cv.Date.max()}', **sp)
  
    k_fold += 1



