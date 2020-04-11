#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
import math

np.random.seed(112)


s0= 40          # Today's stock price
K= 40           # Strike/Exercise price
T= 0.5          # Maturity (in years)
r= 0.05         # Risk-free rate
sigma= 0.2      # Annualized volatility


def Call_option_price(S0, K, T, r, sigma):
    """The call_option_price function calculate the call option prices
 
    Args: 
        S0 (float/int): intial stock price
        K (float/int): strike price
        r (float/int): risk free rate
        T (float/int): term of share price
        sigma (float/int): share volatility

 
    Returns: 
        price (float/int): price of the option
 
    """
    np.random.seed(112)

    n_simulation= 100 # Number of simulations and steps
    n_steps= 100
    dt= T/n_steps

    call= np.zeros([n_simulation], dtype=float)

    for j in range(0,n_simulation):
        sT=S0
        total=0
        for i in range(0,int(n_steps)):
            e = norm.rvs()
            sT*= np.exp((r-0.5*sigma**2)*dt+sigma*e*np.sqrt(dt))
            total+=sT

        price_average = total/n_steps
        call[j]=max(price_average-K, 0)

    price=np.mean(call)*np.exp(-r*T)

    print(f'Call price = {round(price, 3)}')

    return price


Call_option_price(s0,K,T,r,sigma)
