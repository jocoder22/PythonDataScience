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



