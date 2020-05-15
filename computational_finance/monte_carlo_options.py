#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# set stock parameters
stockprice = 100
strikeprice = 100
annualized_vol = 0.15   # implied annualised volatility
ndays = 252

# set monte carlo parameters
np.random.seed(0)
nsimulations = 500000

# estimate the returns
returns_ = np.random.randn(nsimulations, ndays)*annualized_vol/np.sqrt(ndays)

# calcuate teh stock price paths
stockpath = np.cumprod(1+returns_, axis=1)*stockprice

# plot 50 first returns paths
plt.figure(figsize=[10,8])
plt.plot(stockpath[:50,:])
plt.xlabel("Days", fontsize=11)
plt.ylabel("Stock Price", fontsize=11)
plt.show()


# option prices: vanilla options
# call options
call_price = np.mean((stockpath[:,-1] - strikeprice)*((stockpath[:,-1] - strikeprice) > 0))
put_price = np.mean((strikeprice - stockpath[:,-1])*((stockpath[:,-1] - strikeprice) < 0))

