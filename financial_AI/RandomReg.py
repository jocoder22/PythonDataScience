#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr

from datetime import datetime, date

from sklearn.ensemble import RandomForestRegressor as regg
from sklearn.model_selection import train_test_split

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
