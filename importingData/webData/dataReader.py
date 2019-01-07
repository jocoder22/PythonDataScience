#!/usr/bin/env python
## https://pandas-datareader.readthedocs.io/en/latest/index.html

import datetime
import pandas_datareader as pdr

symbol = 'AAPL'
starttime = datetime.datetime(2015, 1, 1)
endtime = datetime.datetime(2018, 12, 31)
apple = pdr.get_data_yahoo(symbol, starttime, endtime)
type(apple)
# <class 'pandas.core.frame.DataFrame'>
apple.to_csv('apple.csv')

symbol2 = '^GSPC'
starttime2 = datetime.datetime(2010, 1, 1)
endtime2 = datetime.datetime(2019, 1, 7)
sp500 = pdr.get_data_yahoo(symbol2, starttime2, endtime2)
sp500.to_csv('sp500.csv')

