import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
from pandas.util.testing import assert_frame_equal


stocklist = ["C","JPM","MS", "GS"]

starttime = datetime.datetime(2000, 1, 1)
portfolio = pdr.get_data_yahoo(stocklist, starttime)

# multiindex dataframe
print(portfolio.head(), portfolio.columns, portfolio.info())

# get only the closing prices
# portfolio_close = pdr.get_data_yahoo(stocklist, starttime)['Close']
portfolio_close = portfolio['Close']

print2(portfolio_close.head())