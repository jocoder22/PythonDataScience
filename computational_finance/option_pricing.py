#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, date, timedelta

# from iexfinance.refdata import get_symbols
# from iexfinance.stocks import Stock, get_historical_intraday, get_historical_data

# pathtk = r"D:\PPP"
# sys.path.insert(0, pathtk)

# import wewebs


# def print2(*args):
#     for arg in args:
#         print(arg, sep="\n\n", end="\n\n")


# sp = {'sep': '\n\n', 'end': '\n\n'}

# path = r"D:\Intradays"

# ttt = wewebs.token

# stock = "NFLX"

# startdate = datetime(2016, 2, 2)
# enddate = datetime(2018, 5, 30)
# stdate = date.today() - timedelta(days=456)


# # allstocks = pdr.get_data_yahoo(stock, startdate)['Adj Close']
# # print(allstocks.head())
# get_symbols(output_format='pandas', token=ttt)

# neflex = Stock(stock, token=ttt)
# print2(neflex.get_quote()['close'])

# start = datetime(2017, 1, 1)
# end = datetime(2018, 1, 1)

# df = get_historical_data("TSLA", start, end, token=ttt, output_format='pandas')

# print2(df.close.var())


startdate = datetime(2013, 2, 2)
# enddate = datetime(2018, 5, 30)

stock = 'NFLX'

allstocks = pdr.get_data_yahoo(stock, startdate)
# Compute the logarithmic returns using the Closing price 
allstocks['Log_Ret'] = np.log(allstocks['Close'] / allstocks['Close'].shift(1))

# Compute Volatility using the pandas rolling standard deviation function
allstocks['Volatility'] = allstocks['Log_Ret'].rolling(window=252).std() * np.sqrt(252)

# Plot the NIFTY Price series and the Volatility
allstocks[['Close', 'Volatility']].plot(subplots=True, color='blue',figsize=(8, 6))
plt.show()

print(allstocks.head())



def sharpe(returns, rf, days=252):
    volatility = returns.std() * np.sqrt(days) 
    sharpe_ratio = (returns.mean() - rf) / volatility
    return sharpe_ratio



def information_ratio(returns, benchmark_returns, days=252):
    return_difference = returns - benchmark_returns 
    volatility = return_difference.std() * np.sqrt(days) 
    information_ratio = return_difference.mean() / volatility
    return information_ratio


def modigliani_ratio(returns, benchmark_returns, rf, days=252):
    volatility = returns.std() * np.sqrt(days) 
    sharpe_ratio = (returns.mean() - rf) / volatility 
    benchmark_volatility = benchmark_returns.std() * np.sqrt(days)
    m2_ratio = (sharpe_ratio * benchmark_volatility) + rf
    return m2_ratio