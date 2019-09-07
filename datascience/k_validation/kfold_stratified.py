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

