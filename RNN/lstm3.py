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
data = rel[['Close']].values
scaler = MinMaxScaler()
datas = scaler.fit_transform(data.reshape(-1, 1))
print(datas[:6])

def pppp(dat, w):
    x = []
    y = []
    for i in range(w, len(dat)):
        x.append(dat[i-w:i, 0])
        y.append(dat[i, 0])

    return np.array(x).reshape(-1, w), np.array(y).reshape(-1)


def train_validate_test_split2(datatt, tx, vx):
    datatt = pd.DataFrame(datatt)
    ww = datatt.shape[1]
    vxx = tx + vx
    test, validate, train = np.split(
        datatt.sample(frac=1), [int(tx*len(datatt)), int(vxx*len(datatt))])

    return np.array(train), np.array(validate), np.array(test)
    # return np.array(train).reshape(-1, ww, 1), np.array(validate).reshape(-1, ww, 1),np.array(test).reshape(-1, ww, 1) 


wd = 60 
val = 0.1
test = 0.1
x, y = pppp(datas, wd)


xtrain, xval, xtest = train_validate_test_split2(x, val, test)
ytrain, yval, ytest = train_validate_test_split2(y, val, test)

xtrain, xval, xtest = xtrain.reshape(-1, wd, 1), xval.reshape(-1, wd, 1), xtest.reshape(-1, wd, 1)

ytrain, yval, ytest = ytrain.reshape(-1, 1), yval.reshape(-1, 1), ytest.reshape(-1, 1)

print(xtrain.shape, xval.shape, xtest.shape, sep=sp)
print(ytrain.shape, yval.shape, ytest.shape, sep=sp)


# build the model
model = Sequential()
model.add(LSTM(50, return_sequences=True,
                       input_shape=(len(xtrain[0]), 1)))
# model.add(Dropout(0.2))

# model.add(LSTM(150, return_sequences=True))
# model.add(Dropout(0.2))

model.add(LSTM(5))

model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='Adam')

model_history = model.fit(
    xtrain, ytrain, epochs=30, batch_size=100, verbose=1, validation_data=(xval, yval), shuffle=False)

lossValues = pd.DataFrame(model.history.history)
lossValues = lossValues.rename({'val_loss': 'ValidationLoss',  'val_acc': 'Val_Accuray',
                                'loss': 'TrainLoss', 'acc': 'TrainAccuracy'}, axis='columns')


# plot loss values
plt.plot(lossValues['ValidationLoss'])
plt.plot(lossValues['TrainLoss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Loss curve')
plt.legend(['Validation Loss', 'Train Loss'])
plt.show()


print(model.summary())


ypred1 = model.predict(xtest)
ypred = scaler.inverse_transform(ypred1)

plt.plot(ypred)
plt.plot(scaler.inverse_transform(ytest))
plt.xlabel('Time')
plt.ylabel('Stock Close')
plt.title('Prediction vs Actual')
plt.legend(['Prediction', 'Actual'])
plt.show()


modelAnalysis = np.sqrt(np.mean(
    np.power((ypred - scaler.inverse_transform(ytest)), 2)))

print(modelAnalysis)
