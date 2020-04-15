#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import norm


np.random.seed(112)

# market information
r= 0.1           # Risk-free rate

# share specific information
s0= 100          # Today's stock price
sigma= 0.3       # Annualized volatility

# call option specific information
K= 110           # Strike/Exercise price
T= 1             # Maturity (in years)

# firm specific information
v0 = 200                # firm current value
sigma_firm = 0.25       # firm volatility 
debt = 180              # firm debt
recovery_rate = 0.2     # recovery rate




# 1. Write a function which takes a risk-free rate, the initial share price, the share volatility, 
# and term as inputs, and determines the terminal value of a share price, 
# assuming geometric Brownian Motion. Note, you should vectorize this function where possible. 

def terminal_shareprice(present_price, risk_free, sigma, Z, T):
    """ terminal_shareprice function gives the terminal value of a share price,
        assuming geometric Brownian Motion and vectorization where possible.
        
    Inputs: 
        present_price(float/int): initial share price
        riskfree(float/int): risk free rate
        sigma: share volatility
        Z: normal random variables
        T(float/int): term of share price
        
    Output:
        terminal value of a share price
    
    """
    
    return present_price*np.exp((risk_free - sigma**2/2)*T + sigma*np.sqrt(T)*Z)