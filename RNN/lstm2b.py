#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import sklearn
from datetime import datetime, date
import tensorflow as tf

import talib as tb
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, Dropout, Flatten
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


import pandas_datareader as pdr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = r'C:\Users\Jose\Desktop\PythonDataScience\RNN\model2'   
os.chdir(path)
savedir = os.path.join(os.getcwd(), 'model2')

def data_normalizer(data_t):
    global scaler_x 
    scaler_x = MinMaxScaler()
    for i in data_t.columns.tolist():
        data_t[i] = scaler_x.fit_transform(data_t[i].values.reshape(-1,1))
    return data_t


sp = '\n\n'
# symbol = 'AAL'
symbol = 'RELIANCE.NS'
starttime = datetime(1996, 1, 1)
endtime = date.today()
rel = pdr.get_data_yahoo(symbol, starttime, endtime)


rel['H-L'] = rel['High'] - rel['Low']
rel['MidHL'] = (rel['High'] + rel['Low'])/ 2
rel['O-C'] = rel['Close'] - rel['Open']
# rel['3day MA'] = rel['Close'].shift(1).rolling(window=3).mean()
# rel['10day MA'] = rel['Close'].shift(1).rolling(window=10).mean()
# rel['30day MA'] = rel['Close'].shift(1).rolling(window=30).mean()
# rel['7dayvol_mean'] = rel['Volume'].shift(1).rolling(window=7).mean()
# rel['Std_dev'] = rel['Close'].rolling(5).std()
# RSI(rel, 'Adj Close', 9)
# rel['RSI'] = tb.RSI(rel['Close'].values, timeperiod=9)
# rel['Williams %R'] = tb.WILLR(
#     rel['High'].values, rel['Low'].values, rel['Close'].values, 7)
# rel['Close2'] = rel['Close']

rel = rel.dropna()
rel = rel.drop(columns=['High', 'Low', 'Volume', 'Open', 'Adj Close'])
print(rel.head(), end=sp)

relnorm = rel.copy()
relnorm.reset_index(drop=True, inplace=True)
relnorn = data_normalizer(relnorm)
print(relnorm.head(), relnorm.shape, relnorm.columns, sep=sp)


# spliting data
vsize_percent = 10
tsize_percent = 10
slen = 15

def lldata(dtaa, sslen):
    data2 = dtaa.values
    data =[]
    for idx in range(len(data2) - sslen):
        data.append(data2[idx: idx + sslen])
    data = np.array(data)
    vsize = int(np.round(vsize_percent / 100 * data.shape[0]))
    tsize = int(np.round(tsize_percent / 100 * data.shape[0]))
    train = data.shape[0] - (vsize + tsize)
    x_train = data[:train,:-1, :]
    y_trian = data[:train, -1, :]
    x_valid = data[train: train + vsize, :-1, :]
    y_valid = data[train: train + vsize, -1, :]
    x_test = data[train + vsize:, :-1, :]
    y_test = data[train + vsize:, -1, :]
    return [x_train, y_trian, x_valid, y_valid, x_test, y_test]

x_train, y_train, x_valid, y_valid, x_test, y_test = lldata(relnorm, slen)

print('x_train.shape = ', x_train.shape, '   x_valid.shape = ', x_valid.shape, '  x_test.shape = ',x_test.shape, end=sp, sep ="")
print('y_trian.shape = ',y_train.shape, '   y_valid.shape = ', y_valid.shape, '   y_test.shape = ', y_test.shape, end=sp, sep="")

# create parameters and placeholders
nsteps = slen - 1
ninput = rel.shape[1]
nneurons = 200
nlayers = 2
lrate = 0.001
batchsize = 60
nepochs = 500
trainsize = x_train.shape[0]
testsize = x_test.shape[0]
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, nsteps, ninput])
y = tf.placeholder(tf.float32, [None, ninput])

index_epoch = 0 
permArray = np.arange(x_train.shape[0])
np.random.shuffle(permArray)

def iter_batch(batch_size):
    global index_epoch, x_train, permArray
    start = index_epoch
    index_epoch += batch_size
    if index_epoch > x_train.shape[0]:
        np.random.shuffle(permArray)
        start = 0
        index_epoch = batch_size
    end = index_epoch
    return x_train[permArray[start:end]], y_train[permArray[start:end]]

