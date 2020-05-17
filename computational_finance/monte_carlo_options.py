#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def pp2(*args):
  for arg in args:
    print(arg, end="\n\n")

# set stock parameters
stockprice = 100
strikeprice = 100
annualized_vol = 0.15   # implied annualised volatility
r = 0.02
ndays = 252


# set monte carlo parameters
np.random.seed(0)
nsimulations = 1000000

# estimate the returns
returns_ = np.random.randn(nsimulations, ndays)*annualized_vol/np.sqrt(ndays)

# calcuate teh stock price paths
stockpath = np.cumprod(1+returns_, axis=1)*stockprice

# plot 50 first returns paths
plt.figure(figsize=[10,8])
plt.plot(stockpath[:5,:])
plt.xlabel("Days", fontsize=11)
plt.ylabel("Stock Price", fontsize=11)
plt.show()


# option prices: vanilla options
# call options
call_price = np.mean((stockpath[:,-1] - strikeprice)*((stockpath[:,-1] - strikeprice) > 0))
put_price = np.mean((strikeprice - stockpath[:,-1])*((stockpath[:,-1] - strikeprice) < 0))
pp2(call_price, put_price)


# include risk free rate
# include risk free rate
def option_price(S,K,T,sigma,rate, type="call", N=10000):
  discount = np.exp(-rate*(T/252))
  _price = np.cumprod(1+ (np.random.randn(T,N) *sigma/np.sqrt(252)), axis=0)*S
  if type == "call":
    return np.sum((_price[-1,:] - K*discount)[_price[-1,:] > K*discount])/_price.shape[1]
  else:
    return -np.sum((_price[-1,:] - K*discount)[_price[-1,:] < K*discount])/_price.shape[1]

days = 126
call2 = option_price(stockprice,strikeprice,days,annualized_vol,r)
put2 = option_price(stockprice,strikeprice,days,annualized_vol,r, type="put")
pp2(call2, put2)
