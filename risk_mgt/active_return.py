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

