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


def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")


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




import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt
import math
import random



#Code to estimate the value of the cos integral
random.seed(0)

mcos_estimates = [None]*50
mcos_std = [None]*50

for i in range(1,51):
    unif_array = uniform.rvs(size = 1000*i)*2
    mcos_val = np.cos(unif_array)*2
    mcos_estimates[i-1] = np.mean(mcos_val)
    mcos_std[i-1] = np.std(mcos_val)/np.sqrt(1000*i)
    
#For the analytic solution
mcos_analytic = np.sin(2) - np.sin(0)

#Plotting the graphs
plt.plot([mcos_analytic]*50)
plt.plot(mcos_estimates,'.')
plt.plot(mcos_analytic+np.array(mcos_std)*3, 'r')
plt.plot(mcos_analytic-np.array(mcos_std)*3, 'r')
plt.xlabel("Sample Size")
plt.ylabel("Value")
plt.show()


# In[5]:


#Code for the put option
random.seed(0)

#Share information
sigma = 0.3
r = 0.1
S0 = 100

#Option information
T = 0.5
K = 110

#Function for terminal share valuation
def terminal_shareprice(S_0, risk_free_rate,sigma,Z,T):
    """Generates the terminal share price given some random normal values, Z"""
    return S_0*np.exp((risk_free_rate-sigma**2/2)*T+sigma*np.sqrt(T)*Z)

#Function for put valuations
def put_price(S_0,K,risk_free_rate,sigma,Z,T):
    """Function for evaluating the call price in Monte Carlo Estimation"""
    share_terminal = terminal_shareprice(S_0, risk_free_rate, sigma, Z, T)
    return np.exp(-risk_free_rate*T)*np.maximum(K-share_terminal,0)

#Empty vectors to be filled later
mput_estimates = [None]*50
mput_std = [None]*50

#Applying MC estimation
for i in range(1,51):
    norm_array = norm.rvs(size = 1000*i)
    mput_val = put_price(S0,K,r,sigma,norm_array,T)
    mput_estimates[i-1] = np.mean(mput_val)
    mput_std[i-1] = np.std(mput_val)/np.sqrt(1000*i)
    
#Determining the analytical solution
d_1 = (math.log(S0/K)+(r + sigma**2/2)*T)/(sigma*math.sqrt(T))
d_2 = d_1 - sigma*math.sqrt(T)
mput_analytic = K*math.exp(-r*T)*norm.cdf(-d_2)-S0*norm.cdf(-d_1)

print2(mput_analytic, mput_estimates[49])

#Plotting the graph
plt.plot([mput_analytic]*50)
plt.plot(mput_estimates,'.')
plt.plot(mput_analytic+np.array(mput_std)*3, 'r')
plt.plot(mput_analytic-np.array(mput_std)*3, 'r')
plt.xlabel("Sample Size")
plt.ylabel("Value")


