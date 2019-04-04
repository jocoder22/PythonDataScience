#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr

from datetime import datetime, date

from sklearn.ensemble import RandomForestRegressor as regg
from sklearn.model_selection import train_test_split as tts

plt.style.use('ggplot')

sp = '\n\n'

start = datetime(2010, 6, 29)
end = datetime(2018, 3, 27)
symbol = 'TSLA'

stock = pdr.get_data_yahoo(symbol, start, end)[['Close']]

for d in range(1, 41):
    dd = 'day' + str(d)
    stock[dd] = stock['Close'].shift(-1 * d)

stock.dropna(inplace=True)

print(stock.head(), stock.tail(), sep=sp, end=sp)

X = stock.iloc[:, :33]
y = stock.iloc[:, 33:]

print(X.shape, y.shape, sep=sp, end=sp)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3)

regressor = regg(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=16,
           oob_score=True, random_state=None, verbose=1, warm_start=False)
