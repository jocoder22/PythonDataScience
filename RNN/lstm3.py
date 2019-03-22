#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from datetime import datetime, date
# import tensorflow as tf

import talib as tb
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, SimpleRNN, Dropout, Flatten
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

import pandas_datareader as pdr
import seaborn as sns

import matplotlib
# print(matplotlib.style.available)
# print(matplotlib.style.library)
plt.style.use('dark_background')

sp = '\n\n'
symbol = 'RELIANCE.NS'
starttime = datetime(1996, 1, 1)
endtime = date.today()
rel = pdr.get_data_yahoo(symbol, starttime, endtime)
rel = rel.drop(columns=['Volume', 'Adj Close'])
print(rel.head(), rel.shape, sep=sp)


rel["Close"].plot(label="Close Price")
rel["High"].plot(label="High Price", color='green')
plt.legend()
plt.show()

rel.reset_index(drop=True, inplace=True)
data = rel[['Close']]
print(data.head())
scaler = MinMaxScaler()
datas = scaler.fit_transform(data)
print(datas[:6])

def pppp(dat, w):
    x = []
    y = []
    for i in range(w, len(dat)):
        x.append(dat[i-w:i, 0])
        y.append(dat[i, 0])

    return np.array(x), np.array(y)


def train_validate_test_split2(datatt, tx, vx):
    vxx = tx + vx
    train, validate, test = np.split(
        datatt.sample(frac=1), [int(tx*len(datatt)), int(vxx*len(datatt))])

    return train, validate, test


w = 60 
x, y = pppp(datas, w)
