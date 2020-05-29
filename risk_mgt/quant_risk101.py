import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
from pandas.util.testing import assert_frame_equal

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

stocklist = ["C","JPM","MS", "GS"]
p_labels = ["Citibank", "J.P. Morgan", "Morgan Stanley", "Goldman Sachs"]

starttime = datetime.datetime(2000, 1, 1)
portfolio = pdr.get_data_yahoo(stocklist, starttime)

# multiindex dataframe
print2(portfolio.head(), portfolio.columns, portfolio.info())

# get only the closing prices
# portfolio_close = pdr.get_data_yahoo(stocklist, starttime)['Close']
portfolio_close = portfolio['Close']
portfolio_close.columns = p_labels
print2(portfolio_close.head())

# Plot portfolio's asset prices
portfolio_close.plot().set_ylabel("Closing Prices, USD")
plt.show()


# Select portfolio asset prices for the middle of the crisis, 2008-2009
asset_prices = portfolio_close.loc['2008':'2009']

# Plot portfolio's asset prices during this time
asset_prices.plot().set_ylabel("Closing Prices, USD")
plt.show()