#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce
from operator import mul

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import linear_model
from sklearn.decomposition import PCA

pd.core.common.is_list_like = pd.api.types.is_list_like

import pandas_datareader as pdr
from pandas_datareader import data as pdrdd
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

import holoviews as hv
import hvplot
import hvplot.pandas


from printdescribe import print2

hv.extension('bokeh')
np.random.seed(42)

# apple = pdr.robinhood.RobinhoodHistoricalReader(['AAPL'], retry_count=3, pause=0.1,
#                                                 timeout=30, session=None, interval='day',
#                                                 span='year').read().reset_index()

# dw = durbin_watson(pd.to_numeric(apple.close_price).pct_change().dropna().values)
# print2(f'DW_Statistics: {dw}')

# Define start and end dates
starttime = '2018-01-01'
endtime = '2019-01-01'

# Download apple stock prices
apple =  pdr.get_data_yahoo('AAPL', starttime, endtime)
dw = durbin_watson(pd.to_numeric(apple.Close).pct_change().dropna().values)

# Compute durbin_watson statistics
print2(f'DW_Statistics: {dw}')

# Get Nasdaq tickers
tickers = pdr.nasdaq_trader.get_nasdaq_symbols(retry_count=3, timeout=300, pause=None)
etfs = tickers.loc[tickers.ETF == True, :]
symbols = etfs.sample(75).index.tolist()
print2(etfs.head(), etfs.shape, symbols)

all_symbols = pdr.get_nasdaq_symbols()
print2(all_symbols.head())


# Get the symbols directly
symbols = get_nasdaq_symbols()
print2(symbols.head(), symbols.shape)


# sample Nasdaq tickers
symbols = get_nasdaq_symbols()
etf2 = symbols.loc[symbols.ETF == True, :]
etf_symbols = etf2.sample(75).index.tolist()
print2(etf2.head(), len(etf_symbols))


starttime = '1997-12-31'
endtime = '2018-10-22'

# etfs_data =  pdr.get_data_yahoo(etf_symbols, starttime, endtime)
# etfs_data.dropna(axis=1, inplace=True)
# print2(etfs_data.head(), etfs_data.shape)

import yfinance as yf
data = yf.download(etf_symbols)
data.dropna(axis=1, inplace=True)
print2(data.head())
