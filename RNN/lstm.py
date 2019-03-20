#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import datetime
# import tensorflow as tf

import talib as tb
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, Dropout, Flatten
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


# plt.style.use('ggplot')

def data_normalizer(data_t):
    global scaler_x 
    scaler_x = MinMaxScaler()
    for i in data_t.columns.tolist():
        data_t[i] = scaler_x.fit_transform(data_t[i].values.reshape(-1,1))

    return data_t

path = r'C:\Users\Jose\Desktop\TimerSeriesAnalysis'    
os.chdir(path)

sp = '\n\n'


rel = pd.read_csv('AMZN.csv', parse_dates=True, index_col='Date')
print(rel.head(), end=sp)


rel['H-L'] = rel['High'] - rel['Low']
rel['MidHL'] = (rel['High'] + rel['Low'])/ 2
rel['O-C'] = rel['Close'] - rel['Open']
rel['3day MA'] = rel['Close'].shift(1).rolling(window=3).mean()
rel['10day MA'] = rel['Close'].shift(1).rolling(window=10).mean()
rel['30day MA'] = rel['Close'].shift(1).rolling(window=30).mean()
rel['7dayvol_mean'] = rel['Volume'].shift(1).rolling(window=7).mean()
rel['Std_dev'] = rel['Close'].rolling(5).std()
# RSI(rel, 'Adj Close', 9)
rel['RSI'] = tb.RSI(rel['Close'].values, timeperiod=9)
rel['Williams %R'] = tb.WILLR(
    rel['High'].values, rel['Low'].values, rel['Close'].values, 7)

rel = rel.dropna()
rel = rel.drop(columns=['High', 'Low', 'Volume', 'Open'])
print(rel.head(), end=sp)




print(rel.head(), rel.shape, sep=sp)


traindata = rel[:'2017']
testdata = rel['2017':]

traindata.reset_index(drop=True, inplace=True)
testdata.reset_index(drop=True, inplace=True)

traindata = data_normalizer(traindata)
testdata = data_normalizer(testdata)

print(traindata.head(), testdata.head(), sep=sp)

xtrain = traindata.drop(columns=['Close'])
ytrain = traindata[['Close']]
xtest = testdata.drop(columns=['Close'])
ytest = testdata[['Close']]

print(xtrain.head(), ytrain.head(), xtest.head(), ytest.head(), sep=sp)
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape, sep=sp)


# x = np.array(xtrain).reshape(xtrain.shape[0], xtrain.shape[1], 1)
# y = np.array(ytrain)

# xt = np.array(xtest).reshape(xtest.shape[0], xtest.shape[1], 1)
# yt = np.array(ytest)

# modelRNN = Sequential()
# modelRNN.add(LSTM(250, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
# modelRNN.add(Dropout(0.2))

# modelRNN.add(LSTM(150, return_sequences=True))
# modelRNN.add(Dropout(0.2))

# modelRNN.add(LSTM(50))

# modelRNN.add(Dense(1, activation='linear'))
# modelRNN.compile(loss='mse', optimizer='Adam')

# # path = r'C:\Users\okigboo\Desktop\PythonDataScience\RNN'
# path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\RNN\\"

# os.chdir(path)

# # saving my models
# savedir = os.path.join(os.getcwd(), 'models')
# modelname = 'Best.{epoch:03d}.h5'

# if not os.path.isdir(savedir):
#     os.makedirs(savedir)
# filepath = os.path.join(savedir, modelname)


# # Callbacks
# lrchecker = ReduceLROnPlateau(factor = np.sqrt(0.1), cooldown=0,
#                              patient=3, verbose=1,
#                              min_lr=0.4e-6)

# monitorbest = ModelCheckpoint(filepath=filepath, monitor='val_loss',
#                              verbose=1,
#                              save_best_only=True)

# callbacks = [lrchecker, monitorbest]
# model_history = modelRNN.fit(
#     x, y, epochs=50, batch_size=50, verbose=1, validation_split=0.2, callbacks=callbacks)


# lossValues = pd.DataFrame(modelRNN.history.history)
# # lossValues = modelRNN.history
# lossValues = lossValues.rename({'val_loss': 'ValidationLoss',  'val_acc': 'Val_Accuray',
#                                 'loss': 'TrainLoss', 'acc': 'TrainAccuracy'}, axis='columns')


# # plot loss values
# plt.plot(lossValues['ValidationLoss'])
# plt.plot(lossValues['TrainLoss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.yscale('log')
# plt.title('Loss curve')
# plt.legend(['Validation Loss', 'Train Loss'])
# plt.show()


# print(modelRNN.summary())


# ypred = modelRNN.predict(xt)
# ypred = scaler_x.inverse_transform(ypred)

# plt.plot(ypred)
# plt.plot(scaler_x.inverse_transform(yt))
# plt.xlabel('Time')
# plt.ylabel('Stock Close')
# plt.title('Prediction vs Actual')
# plt.legend(['Prediction', 'Actual'])
# plt.show()


# modelAnalysis = np.sqrt(np.mean(
#     np.power((ypred - scaler_x.inverse_transform(yt)), 2)))

# print(modelAnalysis)


 

# #################################


def pppp(xd, yd, w):
    x = []
    y = []
    for i in range(w, len(xd)):
        x.append(xd.iloc[i- w:i, :].values.flatten().tolist())
        y.append(yd.iloc[i,:])

    return np.array(x), np.array(y)

window = 60
xt2, yt2 = pppp(xtrain, ytrain, window)
xest, yest = pppp(xtest, ytest, window)

xt3 = np.array(xt2).reshape(xt2.shape[0], xt2.shape[1], 1)
yt3 = np.array(yt2)

xt4 = np.array(xest).reshape(xest.shape[0], xest.shape[1], 1)
yt4 = np.array(yest)

print(xt3.shape, yt3.shape, sep=sp)

model2 = Sequential()
model2.add(LSTM(200, return_sequences=True,
                       input_shape=(len(xt3[0]), 1)))
model2.add(Dropout(0.2))

# model2.add(LSTM(100, return_sequences=True))
# model2.add(Dropout(0.2))

model2.add(LSTM(50))

model2.add(Dense(1, activation='linear'))
model2.compile(loss='mse', optimizer='Adam', metrics=['mape'])

model_history = model2.fit(
    xt3, yt3, epochs=20, batch_size=100, verbose=1, validation_split=0.2)


lossValues = pd.DataFrame(model2.history.history)
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


print(model2.summary())


ypred2 = model2.predict(xt4)
ypred2 = scaler_x.inverse_transform(ypred2)

plt.plot(ypred2)
plt.plot(scaler_x.inverse_transform(yt4))
plt.xlabel('Time')
plt.ylabel('Stock Close')
plt.title('Prediction vs Actual')
plt.legend(['Prediction', 'Actual'])
plt.show()


modelAnalysis = np.sqrt(np.mean(
    np.power((ypred2 - scaler_x.inverse_transform(yt4)), 2)))

print(modelAnalysis)

