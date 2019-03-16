#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime
import tensorflow as tf

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
enddate = datetime(2019, 3, 16)

stock = pdr.get_data_yahoo(stocksname, startdate, enddate)


# Feature engineering
stock['MidPrice'] = (stock.High + stock.Low) / 2.0
stock.reset_index(inplace=True)
# print(stock.head(), stock.tail(), stock.shape, sep=sp)

# EDA:
# print(stock.isnull().sum())
# print(stock.info(), stock.describe(), sep=sp)

stock['MidPrice'].plot()
plt.show()

train_data = stock.loc[:2770, 'MidPrice'].values
test_data = stock.loc[2770:, 'MidPrice'].values

print(type(train_data), test_data.shape)

scalerMM = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)
print(train_data.shape, test_data.shape, train_data.size)

print(train_data[:5])
print(train_data[-5:])
smoothing_window_size = 500 # around 25% of the size
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