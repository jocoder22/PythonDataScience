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


kf_object = KFold(n_splits=5, shuffle=True, random_state=1973)

k_fold = 0
for train_idx, test_idx in kf_object.split(df):
    train_cv, test_cv = df.iloc[train_idx], df.iloc[test_idx]
    print(f'Fold: {k_fold}')
    print(f'Train fold shape: {train_cv.shape}')
    print(f'Test fold shape: {test_cv.index(test_cv.iloc[-1])}', **sp)
    k_fold += 1


    