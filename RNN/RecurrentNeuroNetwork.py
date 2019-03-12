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
    train, validate, test = np.split(
        datatt.sample(frac=1), [int(tx*len(datatt)), int(vxx*len(datatt))])

    return train, validation, test


def data_normalizer(data_t):
    global scaler_x 
    scaler_x = MinMaxScaler()
    for i in data_t.columns.tolist():
        data_t[i] = scaler_x.fit_transform(data_t[i].values.reshape(-1,1))

    return data_t


sp = '\n\n'
symbol = 'RELIANCE.NS'
starttime = datetime.datetime(1996, 1, 1)
endtime = datetime.datetime(2019, 3, 8)
rel = pdr.get_data_yahoo(symbol, starttime, endtime)[['Open','High', 'Low', 'Close', 'Volume']]
print(rel.head(), end=sp)


# https://www.youtube.com/watch?v=dNFgRUD2w68

# Visualizations

rel['H-L'] = rel['High'] - rel['Low']
rel['O-C'] = rel['Close'] - rel['Open']
rel['3day MA'] = rel['Close'].shift(1).rolling(window=3).mean()
rel['10day MA'] = rel['Close'].shift(1).rolling(window=10).mean()
rel['30day MA'] = rel['Close'].shift(1).rolling(window=30).mean()
rel['7dayvol_mean'] = rel['Volume'].shift(1).rolling(window=7).mean()
rel['Std_dev'] = rel['Close'].rolling(5).std()

rel = rel.dropna()
rel = rel.drop(columns=['Open', 'High', 'Low', 'Volume'])
print(rel.head(), end=sp)
# rel['RSI'] = talib.RSI(rel['Close'].values, timeperiod=9)
# rel['Williams %R'] = talib.WILLR(
#     rel['High'].values, rel['Low'].values, rel['Close'].values, 7)

# pd.plotting.scatter_matrix(rel, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
# plt.show()
# sns.pairplot(rel)
# plt.show()

# corr = rel.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=False)
# plt.show()

# # Scale data
# scaler = MinMaxScaler()
# standscaler = StandardScaler()

# # Because the distribution does not approx normal, the MinMaxScaler will be better
# mmdata = scaler.fit_transform(rel)
# staddata = standscaler.fit_transform(rel)

# X = rel.drop('Close', axis=1)
# y = rel[['Close']]
# # Create training, validataion and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42)


data_normalizer(rel)
print(rel.head(), rel.shape, sep=sp)


traindata = rel[:'2017']
testdata = rel['2017':]

traindata.reset_index(drop=True, inplace=True)
testdata.reset_index(drop=True, inplace=True)

print(traindata.head(), testdata.head(), sep=sp)

xtrain = traindata.drop(columns=['Close'])
ytrain = traindata[['Close']]
xtest = testdata.drop(columns=['Close'])
ytest = testdata[['Close']]

print(xtrain.head(), ytrain.head(), xtest.head(), ytest.head(), sep=sp)
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape, sep=sp)


x = np.array(xtrain).reshape(xtrain.shape[0], xtrain.shape[1], 1)
y = np.array(ytrain)

xt = np.array(xtest).reshape(xtest.shape[0], xtest.shape[1], 1)
yt = np.array(ytest)

modelRNN = Sequential()
modelRNN.add(SimpleRNN(50, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
modelRNN.add(Dropout(0.2))

modelRNN.add(SimpleRNN(50, return_sequences=True))
modelRNN.add(Dropout(0.2))

modelRNN.add(SimpleRNN(50))

modelRNN.add(Dense(1, activation='linear'))
modelRNN.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

model_history = modelRNN.fit(
    x, y, epochs=30, batch_size=50, verbose=1, validation_split=0.2)


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


ypred = modelRNN.predict(xt)
ypred = scaler_x.inverse_transform(ypred)

plt.plot(ypred)
plt.plot(scaler_x.inverse_transform(yt))
plt.xlabel('Time')
plt.ylabel('Stock Close')
plt.title('Prediction vs Actual')
plt.legend(['Prediction', 'Actual'])
plt.show()


modelAnalysis = np.sqrt(np.mean(
    np.power((ypred - scaler_x.inverse_transform(yt)), 2)))

print(modelAnalysis)


 

#################################

""" 
def pppp(xd, yd, w):
    x = []
    y = []
    for i in range(w, len(xd)):
        x.append(xd.iloc[i- w:i, :].values.flatten().tolist())
        y.append(yd.iloc[i,:])

    return np.array(x), np.array(y)

window = 20
xt2, yt2 = pppp(xtrain, ytrain, window)
xest, yest = pppp(xtest, ytest, window)

xt3 = np.array(xt2).reshape(xt2.shape[0], xt2.shape[1], 1)
yt3 = np.array(yt2)

xt4 = np.array(xest).reshape(xest.shape[0], xest.shape[1], 1)
yt4 = np.array(yest)


model2 = Sequential()
model2.add(SimpleRNN(20, return_sequences=True,
                       input_shape=(len(xt3[0]), 1)))
model2.add(Dropout(0.2))

model2.add(SimpleRNN(10, return_sequences=True))
model2.add(Dropout(0.2))

model2.add(SimpleRNN(5))

model2.add(Dense(1, activation='linear'))
model2.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

model_history = model2.fit(
    xt3, yt3, epochs=20, batch_size=50, verbose=1, validation_split=0.2)


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
 """