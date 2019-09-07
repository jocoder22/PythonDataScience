#!/usr/bin/env python
import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
from kfold_nonserial import changepath

path = 'C:\\Users\\Jose\\Desktop\\TimerSeriesAnalysis'
sp = {'sep':'\n\n', 'end':'\n\n'}


with changepath(path):
    df = pd.read_csv('AMZN.csv')

df['Date'] = pd.to_datetime(df['Date'])
df["year"] =  df.Date.dt.year

print(df.head(), df.info(), **sp)
print(df.year.value_counts(), **sp)


kf_stratified_object = StratifiedKFold(n_splits=5, shuffle=True, random_state=1973)

k_fold = 0
for train_idx, test_idx in kf_stratified_object.split(df, df['year']):
    train_cv, test_cv = df.iloc[train_idx], df.iloc[test_idx]
    print(f'Fold: {k_fold}')
    print(f'Train fold shape: {train_cv.shape}')
    print(f'Train fold range: {train_cv.index.min()} - {train_cv.index.max()}')
    print(f'Test fold shape: {test_cv.shape}, Test fold range: {test_cv.index.min()} - {test_cv.index.max()}', **sp)
  
    k_fold += 1



