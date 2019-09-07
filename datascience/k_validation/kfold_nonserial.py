#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from contextlib import contextmanager

@contextmanager
def changepath(path):
    currentpath = os.getcwd()

    os.chdir(path)

    try:
        yield 

    finally:
        os.chdir(currentpath)


plt.style.use('ggplot')
path = 'C:\\Users\\Jose\\Desktop\\TimerSeriesAnalysis'
sp = {'sep':'\n\n', 'end':'\n\n'}


with changepath(path):
    df = pd.read_csv('AMZN.csv')


print(df.head(), df.info(), **sp)


kf_object = KFold(n_splits=5, shuffle=False, random_state=1973)

k_fold = 0
for train_idx, test_idx in kf_object.split(df):
    train_cv, test_cv = df.iloc[train_idx], df.iloc[test_idx]
    print(f'Fold: {k_fold}')
    print(f'Train fold shape: {train_cv.shape}')
    print(f'Train fold range: {train_cv.index.min()} - {train_cv.index.max()}')
    print(f'Test fold range: {test_cv.index.min()} - {test_cv.index.max()}', **sp)
  
    k_fold += 1


