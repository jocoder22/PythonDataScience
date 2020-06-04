#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
from pandas.util.testing import assert_frame_equal
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.efficient_frontier import EfficientFrontier

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

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


# Compute the annualized average historical return
mu = mean_historical_return(assets, frequency = 252)

# Plot the annualized average historical return
plt.plot(mu, linestyle = 'None', marker = 'o')
plt.show()

# Create the CovarianceShrinkage instance variable
# this is bette, because it shrinks the errors, and give annualized covariance
cs = CovarianceShrinkage(assets).ledoit.wolf()

# Compute the sample covariance matrix of returns
sample_covariance = assets.pct_change().cov() * 252

# Create the EfficientFrontier instance variable
ef = EfficientFrontier(mu, cs)

# Compute the Weights portfolio that maximises the Sharpe ratio
weights = ef.max_sharpe()

# clean_weights() method truncates tiny weights to zero and rounds others
cw = ef.clean_weights()
ef.save_weights_to_file("weights.txt")  # saves to file
print(cw)

# Evaluate performance of optimal portfolio
ef.portfolio_performance(verbose=True)





# Create a dictionary of time periods (or 'epochs')
epochs = { 'before' : {'start': '1-1-2005', 'end': '31-12-2006'},
           'during' : {'start': '1-1-2007', 'end': '31-12-2008'},
           'after'  : {'start': '1-1-2009', 'end': '31-12-2010'}
         }

# Compute the sample covariance matrix of returns
sample_cov = assets.pct_change().cov() * 252

# Compute the returns and efficient covariance for each epoch
e_return = {}
e_cov = {}

for x in epochs.keys():
  sub_price = assets.loc[epochs[x]['start']:epochs[x]['end']]
  e_return[x] = mean_historical_return(assets, frequency = 252)
  e_cov[x] = CovarianceShrinkage(sub_price).ledoit.wolf()

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
