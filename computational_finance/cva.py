#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import progressbar, tqdm
from scipy.stats import norm

def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")


# 1. Write a function which takes a risk-free rate, the initial share price, the share volatility, 
# and term as inputs, and determines the terminal value of a share price, 
# assuming geometric Brownian Motion. Note, you should vectorize this function where possible. 

def terminalValue(present_price, risk_free, sigma, Z, T):
    """ terminalValue function gives the terminal value of a share price,
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


def callpayoff(terminalval, strikeprice):
    """The callpayoff function 
 
    Args: 
        terminalval (float/int): initial share price
        strikeprice (float/int): : strike price
 
    Returns: 
        payoff (float/int)
 
    """
    return np.maximum(terminalval - strikeprice, 0)
    
np.random.seed(0)

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



corr_t = np.linspace(-1 ,1,21)

cva_est = np.zeros(len(corr_t))
cva_std = np.zeros(len(corr_t))



center2 = 0
bar = progressbar.ProgressBar(maxval=200, widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()])
bar.start()
# for i in range(1,200):
#     center2+=1
#     bar.update(center2)
# bar.finish()

numb = 50000

for i in range(len(corr_t)):
    correlation = corr_t[i]

    if (correlation == 1 or correlation == -1 ):
        norm_vec_0 = norm.rvs(size = numb)
        norm_vec_1 = correlation * norm_vec_0
        corr_norm_matrix = np.array([norm_vec_0, norm_vec_1])
    
    else:
        corr_matrix = np.array([[1, correlation], [correlation, 1]])
        norm_matrix = norm.rvs(size = np.array([2, numb]))
        corr_norm_matrix = np.matmul(np.linalg.cholesky(corr_matrix), norm_matrix)

    tem_stock_value = terminalValue(s0, r, sigma, corr_norm_matrix[0,], T)
    call_val = callpayoff(tem_stock_value, K)

    # firm evolution
    term_firm_value = terminalValue(v0, r, sigma_firm, corr_norm_matrix[1,],T)
    amount_lost = np.exp(-r*T)*(1 - recovery_rate)*(term_firm_value < debt)*call_val

    # cva estimation
    cva_est[i] = np.mean(amount_lost)
    cva_std[i] = np.std(amount_lost)/np.sqrt(numb)

    center2+=1
    bar.update(center2)

bar.finish()

print2(cva_est, cva_std)
