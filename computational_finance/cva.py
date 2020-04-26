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


def discounted_call_payoff(S_T, K, risk_free_rate, L, T):
    '''The discounted_call_payoff calculate discounted payoff
    
    Args:
        S_T (float/int): intial stock price
        K (float/int): strike price
        risk_free_rate (float): risk free rate
        L (float/int) : up-and-out barrier
        T (int) : term 
        
    Returns:
        P (float/int):discount option prices
    
    '''
    if (S_T > L):
        return 0
    return np.maximum(S_T-K, 0)


def black_schole_callprice(S, K, T, rf, sigma):
    """The black_schole_callprice function calculates the call option price
        under Black Schole Merton model
 
    Args: 
        S: current stock price
        K: strike price
        T: maturity date in years
        rf: risk-free rate (continusouly compounded)
        sigma: volatiity of underlying security 
 
 
    Returns: 
        callprice: call price
 
    """
    current_time = 0

    d1_numerator = np.log(S/K) + (r + sigma**2/2) * (T - current_time)
    d1_denominator = sigma * np.sqrt(T - current_time)

    d1 = d1_numerator / d1_denominator
    d2 =  d1 - d1_denominator



    callprice = S*norm.cdf(d1) - (norm.cdf(d2)*K*np.exp(-r * (T - current_time)))


    return callprice

    
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

callval_est = np.zeros(len(corr_t))
callval_std = np.zeros(len(corr_t))

callcva_est = np.zeros(len(corr_t))
# callcva_std = np.zeros(len(corr_t))



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
    callval_est[i] = np.mean(call_val)
    callval_std[i] = np.std(call_val)/np.sqrt(numb)

    # firm evolution
    term_firm_value = terminalValue(v0, r, sigma_firm, corr_norm_matrix[1,],T)
    amount_lost = np.exp(-r*T)*(1 - recovery_rate)*(term_firm_value < debt)*call_val

    # cva estimation
    cva_est[i] = np.mean(amount_lost)
    cva_std[i] = np.std(amount_lost)/np.sqrt(numb)

    # calculate option value with cva
    callcva_est[i] = callval_est[i] - cva_est[i]
    # callcva_std[i] = np.sqrt(callval_est[i]**2 + cva_est[i]**2 - 2*np.matmul(corr_norm_matrix,callval_est[i],cva_std[i]))

    center2+=1
    bar.update(center2)

bar.finish()

# calculate firm default probability
d1_numerator = np.log(v0/debt) + (r + sigma_firm**2/2) * T 
d1_denominator = sigma_firm * np.sqrt(T)
d1 = d1_numerator / d1_denominator
d2 =  d1 - d1_denominator

firm_default_prob =  norm.cdf(-d2)

# calculate analytic vanilla European call option price
analytic_callprice = black_schole_callprice(s0,K,T, r, sigma)


# calculate uncorrelated credit valuation adjustment (cva)
uncor_cva = (1 - recovery_rate)*firm_default_prob*analytic_callprice

# plot monte carlo cva estimates for different correlations
plt.plot(corr_t,[uncor_cva]*21)
plt.plot(corr_t, cva_est, ".")
plt.plot(corr_t, cva_est+3*np.array(cva_std), "black")
plt.plot(corr_t, cva_est-3*np.array(cva_std), "g")
plt.title("Monte carlo Credit Valuation Adjustments estimates for different correlations")
plt.xlabel("Correlation")
plt.ylabel("CVA")
plt.show()

corr_t

plt.figure(figsize=[12,8])
plt.plot(corr_t,callval_est, '.')
plt.plot(corr_t, callcva_est,'-')
plt.plot(corr_t,callval_est+3*np.array(callval_std),'black')
plt.plot(corr_t,callval_est-3*np.array(callval_std),'g')

# plt.plot(corr_t,callval_est+3*np.array(callcva_std),'black')
# plt.plot(corr_t,callval_est-3*np.array(callcva_std),'g')

plt.xlabel("Months")
plt.ylabel("Price")
plt.title("Monte Carlo Estimates of risk-adjusted call option price")
plt.legend(('Risk-neutral price', 'Risk-adjusted price', 'Risk-neutral price UB', 'Risk-neutral price LB'))
plt.show()
print2(cva_est, cva_std, firm_default_prob)


url = "https://view98n6mw6nlkh.udacity-student-workspaces.com/edit/data/portfolio.json"
portfolio = pd.read_json(url, orient='records', lines=True)