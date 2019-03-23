#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import datetime
# import talib
# import tensorflow as tf
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, SimpleRNN, Dropout, Flatten


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


import pandas_datareader as pdr
import seaborn as sns

# plt.style.use('ggplot')


def train_validate_test_split(rel, train_percent=.6, validate_percent=.2):
    np.random.seed(3456)
    perm = np.random.permutation(rel.index)
    m = len(perm)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = rel.ix[perm[:train_end]]
    validate = rel.ix[perm[train_end:validate_end]]
    test = rel.ix[perm[validate_end:]]

    return train, validate, test


def train_validate_test_split2(datatt, tx, vx):
    vxx = tx + vx
    train, validation, test = np.split(
        datatt.sample(frac=1), [int(tx*len(datatt)), int(vxx*len(datatt))])

    return train, validation, test


def data_normalizer(data_t):
    global scaler_x
    scaler_x = MinMaxScaler()
    for i in data_t.columns.tolist():
        data_t[i] = scaler_x.fit_transform(data_t[i].values.reshape(-1, 1))

    return data_t


def RSI(dataset, column, peroid):
    # dataset2['closediff'] = dataset2.diff()
    # closediff = closediff[1:].values

    # Make the positive gains (up) and negative gains (down) Series

    # up = closediff[closediff > 0].mean()
    # down = -1 * closediff[closediff < 0].mean()

    dataset['diffa'] = dataset[column].diff()
    dataset['noneg'] = dataset.diffa.where(
        dataset.diffa > 0, 0).shift(1).rolling(window=peroid).mean()
    dataset['neg'] = dataset.diffa.where(dataset.diffa < 0, 0).abs().shift(
        1).rolling(window=peroid).mean()
    
    dataset['RSI'] = 100 * dataset['noneg'] / (dataset['noneg'] + dataset['neg'])
    dataset.drop(columns=['diffa', 'noneg', 'neg'], inplace=True)
    dataset = dataset.dropna()
    return dataset


def RSI2(values):
    up = values[values > 0].mean()
    down = -1*values[values < 0].mean()
    return 100 * up / (up + down)

path = r'C:\Users\okigboo\Desktop\PythonDataScience\RNN'

os.chdir(path)
sp = '\n\n'
symbol = 'RELIANCE.NS'
starttime = datetime.datetime(1996, 1, 1)
endtime = datetime.datetime(2019, 3, 8)
rel = pdr.get_data_yahoo(symbol, starttime, endtime)[['High', 'Close', 'Adj Close']]
rel['AdjDiff'] = rel['Adj Close'].diff()
RSI(rel, 'Adj Close', 9)
rel = rel.dropna()
print(rel.head(20), end=sp)


# https://www.youtube.com/watch?v=dNFgRUD2w68

# Visualizations

# rel['H-L'] = rel['High'] - rel['Low']
# rel['O-C'] = rel['Close'] - rel['Open']
# rel['3day MA'] = rel['Close'].shift(1).rolling(window=3).mean()
# rel['10day MA'] = rel['Close'].shift(1).rolling(window=10).mean()
# rel['30day MA'] = rel['Close'].shift(1).rolling(window=30).mean()
# rel['7dayvol_mean'] = rel['Volume'].shift(1).rolling(window=7).mean()
# rel['Std_dev'] = rel['Close'].rolling(5).std()

# rel = rel.dropna()
# # rel = rel.drop(columns=['Open', 'High', 'Low', 'Volume'])
# # print(rel.head(), end=sp)
# # rel['RSI'] = talib.RSI(rel['Close'].values, timeperiod=9)
# # rel['Williams %R'] = talib.WILLR(
# #     rel['High'].values, rel['Low'].values, rel['Close'].values, 7)


traindata = rel.loc[:'2017', ['RSI']]
testdata = rel.loc['2017':, ['RSI']]
ytraindata = rel.loc[:'2017', ['Close']]
ytestdata = rel.loc['2017':, ['Close']]

traindata.reset_index(drop=True, inplace=True)
testdata.reset_index(drop=True, inplace=True)
ytraindata.reset_index(drop=True, inplace=True)
ytestdata.reset_index(drop=True, inplace=True)

x_train = np.array(traindata)
x_test = np.array(testdata)
y_train = np.array(ytraindata)
y_test = np.array(ytestdata)

# Scale data
scaler = MinMaxScaler()

xtrainscaled = scaler.fit_transform(x_train.reshape(-1, 1))
xtestscaled = scaler.fit_transform(x_test.reshape(-1, 1))
ytrainscaled = scaler.fit_transform(y_train.reshape(-1, 1))
ytestscaled = scaler.fit_transform(y_test.reshape(-1, 1))

# # print(traindata.head(), testdata.head(), sep=sp)

# # xtrain = traindata.drop(columns=['Close'])
# # ytrain = traindata[['Close']]
# # xtest = testdata.drop(columns=['Close'])
# # ytest = testdata[['Close']]



windows = 90
def prepp(datset1, dataset2, windows):
    x = []
    y = []
    for i in range(windows, len(datset1)):
        x.append(datset1[i - windows:i, 0])
        y.append(dataset2[i, 0])

    return np.array(x).reshape(-1, 90, 1), np.array(y).reshape(-1, 1)


x_train_new, y_train_new = prepp(xtrainscaled, ytrainscaled, windows)
x_test_new, y_test_new = prepp(xtestscaled, ytestscaled, windows)

print(traindata.head(), traindata.head(),
      traindata.shape, traindata.shape, sep=sp)

print(x_train_new.shape, y_train_new.shape,
      x_test_new.shape, y_test_new.shape, sep=sp)

print(x_train_new[0], y_train_new[0], sep=sp)



# build the model
modelRNN = Sequential()
modelRNN.add(SimpleRNN(50, return_sequences=True,
                       input_shape=(windows, 1)))
modelRNN.add(Dropout(0.2))

modelRNN.add(SimpleRNN(50, return_sequences=True))
modelRNN.add(Dropout(0.2))

modelRNN.add(SimpleRNN(50))

modelRNN.add(Dense(1, activation='linear'))
modelRNN.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

model_history = modelRNN.fit(
    x_train_new, y_train_new, epochs=30, batch_size=64, verbose=1, validation_split=0.2)


lossValues = pd.DataFrame(modelRNN.history.history)
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


print(modelRNN.summary())


ypred1 = modelRNN.predict(x_test_new)
ypred = scaler.inverse_transform(ypred1)

plt.plot(ypred)
plt.plot(scaler.inverse_transform(y_test_new))
plt.xlabel('Time')
plt.ylabel('Stock Close')
plt.title('Prediction vs Actual')
plt.legend(['Prediction', 'Actual'])
plt.show()


modelAnalysis = np.sqrt(np.mean(
    np.power((ypred - scaler.inverse_transform(y_test_new)), 2)))

print(modelAnalysis)

# save model
modelRNN.save_weights('model_vs1.h5')
