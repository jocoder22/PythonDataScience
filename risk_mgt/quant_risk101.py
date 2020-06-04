#!/usr/bin/env python
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

# set the weights
weights = [0.20, 0.30, 0.30, 0.20]
# Select portfolio asset prices for the middle of the crisis, 2008-2009
asset_prices = portfolio_close.loc['2008':'2009']

# Plot portfolio's asset prices during this time
asset_prices.plot().set_ylabel("Closing Prices, USD")
plt.show()

# calculate percentage return and portfolio return
asset_returns = portfolio_close.pct_change()
portfolio_returns = asset_returns.dot(weights)

# Plot portfolio returns
portfolio_returns.plot().set_ylabel("Daily Return, %")
plt.show()

print2(asset_prices.head(), asset_returns.head(), portfolio_returns.head())

# Generate the covariance matrix from portfolio asset's returns
covariance = asset_returns.cov()

# Annualize the covariance using 252 trading days per year
covariance = covariance * 252

# Display the covariance matrix
print2(covariance)

# Compute and display portfolio volatility
portfolio_variance = np.transpose(weights) @ covariance @ weights
portfolio_volatility = np.sqrt(portfolio_variance)
print2(portfolio_volatility)

# Calculate the 30-day rolling window of portfolio returns
returns_windowed = portfolio_returns.rolling(30)

# Compute the annualized volatility series
volatility_series = returns_windowed.std()*np.sqrt(252)

# Plot the portfolio volatility
volatility_series.plot().set_ylabel("Annualized Volatility, 30-day Window")
plt.show()

