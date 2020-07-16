#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
from pandas.util.testing import assert_frame_equal
from contextlib import contextmanager

from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.cla import CLA
from pypfopt import objective_functions
from pypfopt import Plotting  as pplot

from printdescribe import print2, describe2, changepath


path = r"D:\PythonDataScience\risk_mgt"

stocklist = ["C","JPM","MS", "GS"]
stocklist = ["JPM", "GS", "BAC", "MS", "C","CS",
             "BCS" , "DB", "UBS", "RY", "WFC",
             "HSBC", "JEF", "BNP.PA", "MFG", "LAZ", "NMR", "EVR",
             "BMO", "MUFG"]
                     
p_labels = ["Citibank", "J.P. Morgan", "Morgan Stanley", "Goldman Sachs"]
pp_labels = ["JPMorgan Chase", "Goldman Sachs", "BofA Securities", "Morgan Stanley", "Citigroup", "Credit Suisse", 
             "Barclays Investment Bank", "Deutsche Bank", "UBS", "RBC Capital Markets", "Wells Fargo Securities",
             "HSBC", "Jefferies Group", "BNP Paribas", "Mizuho", "Lazard", "Nomura", "Evercore Partners", 
             "BMO Capital Markets", "Mitsubishi UFJ Financial Group"]

starttime = datetime.datetime(2000, 1, 1)
endtime = datetime.datetime(2019, 10, 1)

# get only the closing prices
assets = pdr.get_data_yahoo(stocklist, starttime, endtime)['Close']


# path2 = r"D:\TimerSeriesAnalysis\AMZN.csv"
path2 = r"D:\TimerSeriesAnalysis"
with changepath(path2):
    df = pd.read_csv("AMZN.csv", parse_dates=True, index_col="Date")

# Compute the annualized average historical return
mu = mean_historical_return(assets, frequency = 252)

# Plot the annualized average historical return
plt.plot(mu, linestyle = 'None', marker = 'o')
plt.show()

# Create the CovarianceShrinkage instance variable
# this is bette, because it shrinks the errors, and give annualized covariance
cs = CovarianceShrinkage(assets).ledoit_wolf()

# Compute the sample covariance matrix of returns
sample_covariance = assets.pct_change().cov() * 252

# Create the EfficientFrontier instance variable
ef = EfficientFrontier(mu, cs)

# Compute the Weights portfolio that maximises the Sharpe ratio
weights = ef.max_sharpe()

# clean_weights() method truncates tiny weights to zero and rounds others
cw = ef.clean_weights()

with changepath(path):
  ef.save_weights_to_file("weights.txt")  # saves to file

print2(cw)

# Evaluate performance of optimal portfolio
ef.portfolio_performance(verbose=True)

# Create a dictionary of time periods (or 'epochs')
epochs = { 'before' : {'start': starttime, 'end': '31-12-2006'},
           'during' : {'start': '1-1-2007', 'end': '31-12-2008'},
           'after'  : {'start': '1-1-2009', 'end': endtime}
         }

# Compute the sample covariance matrix of returns
sample_cov = assets.pct_change().cov() * 252

# Compute the returns and efficient covariance for each epoch
e_return = {}
e_cov = {}

for x in epochs.keys():
  sub_price = assets.loc[epochs[x]['start']:epochs[x]['end']]
  e_return[x] = mean_historical_return(assets, frequency = 252)
  e_cov[x] = CovarianceShrinkage(sub_price).ledoit_wolf()

# Display the efficient covariance matrices for all epochs
print("Efficient Covariance Matrices\n", e_cov)

# Initialize the Crtical Line Algorithm object
efficient_portfolio_during = CLA(e_return["during"], e_cov["during"])

# Find the minimum volatility portfolio weights and display them
print(efficient_portfolio_during.min_volatility())

# Compute the efficient frontier
(ret, vol, weights) = efficient_portfolio_during.efficient_frontier()

# Add the frontier to the plot showing the 'before' and 'after' frontiers
plt.scatter(vol, ret, s = 4, c = 'g', marker = '.', label = 'During')
plt.legend()
plt.show()

# plotting using PyPortfolioOpt
pplot.plot_covariance(cs,plot_correlation=False, show_tickers=True)
pplot.plot_efficient_frontier(efficient_portfolio_during, points=100, show_assets=True)
pplot.plot_weights(cw)

# Dealing with many negligible weights
# efficient portfolio allocation
ef = EfficientFrontier(mu, cs)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)
w = ef.max_sharpe()
print(ef.clean_weights())


# Post-processing weights
# These are the quantities of shares that should be bought to have a $20,000 portfolio
latest_prices = get_latest_prices(assets)
da = DiscreteAllocation(w, latest_prices, total_portfolio_value=20000)
allocation, leftover = da.lp_portfolio()
print2(allocation)

