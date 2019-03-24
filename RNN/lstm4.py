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
from tensorflow.python.keras.models import Sequential, load_model
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
# plt.style.use('dark_background')

path = r'C:\Users\Jose\Desktop\PythonDataScience\RNN'
os.chdir(path)

sp = '\n\n'
symbol = 'AAL'
starttime = datetime(1996, 1, 1)
endtime = date.today()
stock = pdr.get_data_yahoo(symbol, starttime, endtime)
stock.reset_index(inplace=True)
print(stock.head(), stock.shape, sep=sp)

scaler = MinMaxScaler()
closeprice = stock[['Close']]
closeprice = scaler.fit_transform(closeprice)
print(closeprice)

window = 14
val = 0.1
test = 0.1

def preprocess(data, wdw):
    feature, target = [], []
    for idx in range(len(data) - wdw - 1):
        feature.append(data[idx: idx + wdw, 0])
        target.append(data[idx + wdw, 0])

    return np.array(feature), np.array(target)


def train_validate_test_split2(datatt, tx, vx, ww):
    vxx = tx + vx
    test, validate, train = np.split(datatt, [int(tx*len(datatt)), int(vxx*len(datatt))])
    return np.expand_dims(train, axis=-1), np.expand_dims(validate, axis=-1), np.expand_dims(test, axis=-1)

xdata, ydata = preprocess(closeprice, window)

xtrain, xval, xtest = train_validate_test_split2(xdata, val, test, window)
ytrain, yval, ytest = train_validate_test_split2(ydata, val, test, window)

print(xtrain.shape, xval.shape, xtest.shape, sep=sp)
print(ytrain.shape, yval.shape, ytest.shape, sep=sp)


""" now = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
# saving weights
savedir = os.path.join(os.getcwd(), 'weights')
# modelname = 'Best.{epoch:03d}_Loss:{loss:05f}.h5'
modelname = 'Best_{0}.h5'.format(now)
"""


""" 
name_yskd = 900

if not os.path.isdir(savedir):
    os.makedirs(savedir)
filepath = os.path.join(savedir, modelname)

monitorbest = ModelCheckpoint(filepath=filepath, monitor='loss',
                             verbose=1,
                             save_best_only=True)

callbacks=[monitorbest]
""" 
""" 

model = Sequential()
model.add(LSTM(256, input_shape=(window, 1)))
# model.add(Dropout(0.2))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')  
"""

"""


history = model.fit(xtrain, ytrain, epochs=30, validation_data=(xval, yval), 
            callbacks=callbacks, shuffle=False)  """


# model.load_weights('weights\Best.223_0.000160.h5')


model = load_model(f"model\model55.h5")
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.show()
print(model.summary())
pred = model.predict(xtest)
print(type(pred), ytest.shape, sep=sp, end=sp)
# actual, prediction = [], []
prediction = scaler.inverse_transform(pred).reshape(-1)
actual= scaler.inverse_transform(ytest).reshape(-1)
df = pd.DataFrame({'Actual': actual, 'Predictions': prediction})

print(df.shape, df.head(), sep=sp, end=sp)
plt.plot(scaler.inverse_transform(ytest.reshape(-1, 1)), label='Actual', color='red')
plt.plot(scaler.inverse_transform(pred), label='Prediction', color='yellow')
plt.legend()
plt.show()

# now = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
# save model
# model.save_weights('model_lstm.h5')
# model.save(f"model\model55.h5")
