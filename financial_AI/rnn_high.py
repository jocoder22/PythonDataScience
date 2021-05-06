#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, date
import tensorflow as tf

import talib as tb
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, SimpleRNN, Dropout, Flatten
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn import datasets
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
from sklearn.preprocessing import MinMaxScaler
plt.style.use('ggplot')

sp ='\n\n'
stocksname ='AAL'

startdate = datetime(1990, 1, 1)
enddate = date.today()

stock = pdr.get_data_yahoo(stocksname, startdate, enddate)


# Feature engineering
stock['MidPrice'] = (stock.High + stock.Low) / 2.0
allstock = stock.copy()

# print(stock.head(), stock.tail(), stock.shape, sep=sp)
# EDA:
# print(stock.isnull().sum())
# print(stock.info(), stock.describe(), sep=sp)

stock['MidPrice'].plot()
plt.show()

stock.reset_index(inplace=True)
train_data = stock.loc[:2770, 'MidPrice'].values
test_data = stock.loc[2770:, 'MidPrice'].values

print(type(train_data), test_data.shape)

scalerMM = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)
print(train_data.shape, test_data.shape, train_data.size)

print(train_data[:5])
print(train_data[-5:])
smoothing_window_size = 1500 # around 25% of the size
for i in range(0, train_data.shape[0] ,smoothing_window_size):
    window = i+smoothing_window_size
    if window < train_data.shape[0] or window == train_data.shape[0]:
        scalerMM.fit(train_data[i:i+smoothing_window_size,:])
        train_data[i:i+smoothing_window_size,:] = scalerMM.transform(train_data[i:i+smoothing_window_size,:])
    else:
        # # You normalize the last bit of remaining data
        scalerMM.fit(train_data[i:,:])
        train_data[i:,:] = scalerMM.transform(train_data[i:,:])



print(train_data.shape, test_data.shape, train_data.size)
# Reshape both train and test data

train_data = train_data.reshape(-1)
print(train_data[:5])
print(train_data[-5:])

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scalerMM.transform(test_data).reshape(-1)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for i in range(train_data.shape[0]):
  EMA = gamma*train_data[i] + (1-gamma)*EMA
  train_data[i] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)

###############################################################################
# One-Step Ahead Prediction via Averaging
window_size = 20  # around 10% of size
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size, N):

    if pred_idx >= N:
        date = datetime.datetime.strptime(k, '%Y-%m-%d').date() + datetime.timedelta(days=1)
    else:
        date = stock.loc[pred_idx,'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))


std_avg_predictions = scalerMM.inverse_transform(np.array(std_avg_predictions).reshape(-1, 1))
all_mid_data = scalerMM.inverse_transform(all_mid_data.reshape(-1, 1))
data2 = pd.DataFrame(std_avg_predictions, columns=['Prediction'])
data3 = pd.DataFrame(all_mid_data, columns=['Original'])
result = pd.concat([stock, data2, data3], axis=1)

result.set_index('Date', inplace=True)
result.dropna()
print(result.head(), data2.head(), data3.head(), sep=sp)


result['Prediction'].plot(label="Predictions", color='black')
result['Original'].plot(label="Original", color='red')
allstock['MidPrice'].plot(label='Real', color='blue')
plt.legend()
plt.show()

# data1 = np.concatenate(std_avg_predictions, stock)
# Plotting:
plt.plot(all_mid_data, label="Original")
plt.plot(std_avg_predictions, label="Predictions")
plt.legend()
plt.show()





def pppp(xd, yd, w):
    x = []
    y = []
    for i in range(w, len(xd)):
        x.append(xd.iloc[i- w:i, :].values.flatten().tolist())
        y.append(yd.iloc[i,:])
        # x.append(xd[i- w:i, :].values.flatten().tolist())
        # y.append(yd[i,:])

    return np.array(x), np.array(y)

print("########################################", end=sp)
print(train_data.shape, test_data.shape, train_data.size)

windows = 90
def prepp(datset1, windows):
    datset1 = datset1.reshape(-1, 1)
    x = []
    y = []
    for i in range(windows, len(datset1)):
        x.append(datset1[i - windows:i, 0])
        y.append(datset1[i, 0])

    return np.array(x).reshape(-1, 90, 1), np.array(y).reshape(-1, 1)


xt2, yt2= prepp(train_data, windows)
xest, yest = prepp(test_data, windows)
print(xt2.shape, yt2.shape, xest.shape, yest.shape)

# window = 60
# xt2, yt2 = pppp(pd.DataFrame(train_data), pd.DataFrame(test_data), window)
# xest, yest = pppp(xtest, ytest, window)

xt3 = np.array(xt2).reshape(xt2.shape[0], xt2.shape[1], 1)
yt3 = np.array(yt2)

xt4 = np.array(xest).reshape(xest.shape[0], xest.shape[1], 1)
yt4 = np.array(yest)


model2 = Sequential()
model2.add(LSTM(100, return_sequences=True,
                       input_shape=(len(xt3[0]), 1)))
model2.add(Dropout(0.2))

# model2.add(LSTM(100, return_sequences=True))
# model2.add(Dropout(0.2))

model2.add(LSTM(50))

model2.add(Dense(1, activation='linear'))

model2.compile(loss='mse', optimizer='Adam')

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
ypred2 = scalerMM.inverse_transform(ypred2)

plt.plot(ypred2)
plt.plot(scalerMM.inverse_transform(yt4))
plt.xlabel('Time')
plt.ylabel('Stock Close')
plt.title('Prediction vs Actual')
plt.legend(['Prediction', 'Actual'])
plt.show()


modelAnalysis = np.sqrt(np.mean(
    np.power((ypred2 - scalerMM.inverse_transform(yt4)), 2)))

print(modelAnalysis)
