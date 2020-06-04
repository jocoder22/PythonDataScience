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
mean_returns = mean_historical_return(assets, frequency = 252)

# Plot the annualized average historical return
plt.plot(mean_returns, linestyle = 'None', marker = 'o')
plt.show()

# Create the CovarianceShrinkage instance variable
# this is bette, because it shrinks the errors
cs = CovarianceShrinkage(prices).ledoit.wolf()

# Compute the sample covariance matrix of returns
sample_cov = prices.pct_change().cov() * 252

# Create the EfficientFrontier instance variable
ef = EfficientFrontier(mu, S)

# Compute the Weights portfolio that maximises the Sharpe ratio
weights = ef.max_sharpe()

# clean_weights() method truncates tiny weights to zero and rounds others
cw = ef.clean_weights()
ef.save_weights_to_file("weights.txt")  # saves to file
print(cm)

# Evaluate performance of optimal portfolio
ef.portfolio_performance(verbose=True)
