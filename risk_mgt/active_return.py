#!/usr/bin/env python
# Import required modules for this CRT

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
# from statistics import stdev 


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")


# I'm using daily close prices
# SPDR S$p500 ETF (SPY)
SPY = "SPY"

# My two other ETFs are
# 1. Vanguard S$P500 ETF (VOO)
# 2. iShares Core S&P500 ETF (IVV)

etfs_tickers = ["^GSPC", "SPY", "VOO", "IVV"]

# using 2 years of data from January 01, 2018 to December 31, 2019
starttime = datetime.datetime(2018, 1, 1)
endtime = datetime.datetime(2019, 12, 31)

# get only the closing prices
etfs = pdr.get_data_yahoo(etfs_tickers, starttime, endtime)['Close']
etfs.columns = ["S&P500", "SPDR", "Vanguard", "iShares"]

print2(etfs.head())



etfs_return = etfs.pct_change().dropna()

etfs_return.fillna(0, inplace=True)
returns2= round(etfs_return*100, 3)

print2(etfs_return, returns2)


eft_index = etfs_return["S&P500"]
etfs_activeR = etfs_return.sub([eft_index, eft_index,eft_index,eft_index], axis='columns')
etfs_activeR.drop("S&P500", axis=1, inplace=True)
etfs_activeR.columns = ["Active_SPDR", "Active_Vanguard", "Active_iShares"]


r_index = returns2["S&P500"]
r_activeR = returns2.sub([r_index, r_index,r_index, r_index], axis='columns')
r_activeR.drop("S&P500", axis=1, inplace=True)
r_activeR.columns = ["Active_SPDR", "Active_Vanguard", "Active_iShares"]


print2(etfs_activeR, r_activeR)

mean_return = etfs_return.mean()
_return = returns2.mean()


average_return = mean_return - etfs_return["S&P500"].mean()
average_return

cum_return = (1+etfs_return).cumprod() - 1
