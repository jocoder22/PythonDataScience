#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, date, timedelta
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import uniform
import math


# set random seed
np.random.seed(0)

def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")


sp = {'sep': '\n\n', 'end': '\n\n'}

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


#########################################################################################################
#########################################################################################################
# set the seed
np.random.seed(0)

# calculate the analytical values of the integral, anal_val
anal_val, err = quad(lambda x: np.cos(x), 0, 2)
print("Analytical value: ", anal_val)

# we do 50 number of stimulations
num = 50

mt_estimates = [None]*num
mt_std = [None]*num

#  2. Estimate the value of the integral as a function of sample size. 
# Use sample sizes of 1000, 2000, ..., 50000. 
for i in range(1, num+1):
    un_array = uniform.rvs(size = i*1000)*2
    sim_val = np.cos(un_array)*2
    mt_estimates[i-1] = np.mean(sim_val)
    mt_std[i-1] = np.std(sim_val)/np.sqrt(i*1000)
    
print("Monte Carlo estimate: ", mt_estimates[num-1])

# 3. Plot the estimates against the analytical value of the integral.
plt.figure(figsize = [11, 6])
plt.plot([anal_val]*num)
plt.plot(mt_estimates, ".")
plt.plot(anal_val + np.array(mt_std)*3)
plt.plot(anal_val - np.array(mt_std)*3)
plt.fill_between(np.arange(num), anal_val + np.array(mt_std)*3, 
                 anal_val - np.array(mt_std)*3,facecolor='whitesmoke', interpolate=True)
plt.ylabel("Monte carlo estimates")
plt.xlabel("Number of simulations")
plt.title("Monte carlo estimation of integral of cos(X)")
plt.show()




#######################################################################################################
#######################################################################################################
# stock features information
sigma = 0.3
risk_free = 0.1
current_price = 100
T = 0.5

strike_price = 110
current_time = 0


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



# 2. Write a function which takes terminal share prices, a strike price, 
# a risk-free rate and term as inputs, and gives out the discounted value of a European put option. 
def discounted_putpayoff(terminal_price, strikeprice, riskfree, T):
    """ discounted_putpayoff function gives out the discounted value of a European put option.
    
    Inputs: 
        terminal_price(float/int): terminal price of European put option
        strikeprice(float/int): strike price European put option
        riskfree(float/int): risk free rate
        T(float/int): term of European put option
        
    Output:
        discounted value of a European put option
    
    """
    
    return np.exp(-riskfree*(T - current_time))*np.maximum(strikeprice - terminal_price, 0)


numb = 50


# 3. Write a for loop which cycles through sample size (1000, 2000, ..., 50000), and calculates the 
# Monte Carlo estimate of a European put option, and well as the standard deviation of 
# the Monte Carlo estimator.
def option_prices(current_price, risk_free, sigma, term, current_time, type="call", plot=False):
    """The option_prices function calculate both analytical and monte carlo simulation of
        either call or put option
 
    Args: 
        present_price (float/int): initial share price
        riskfree (float/int): risk free rate
        sigma (float/int): share volatility
        Z (float/int): normal random variables
        T (float/int): term of share price
        type (str): type of option, "call" or "put"
        plot (bool): False or True, whether to plot graph or not
 
 
    Returns: 
        price (float/int): price of the option
 
    """
    # number of simulations
    numb = 50

    mput_estimates = np.zeros(numb)
    mput_std = np.zeros(numb)
        

    for i in range(1, numb+1):
        mput_norm = norm.rvs(size=1000*i)
        terminalVals = terminal_shareprice(current_price, risk_free, sigma,mput_norm, T - current_time)
        mputvals = discounted_putpayoff(terminalVals, strike_price, risk_free, T - current_time)
        mput_estimates[i-1] = np.mean(mputvals)
        mput_std[i-1] = np.std(mputvals)/np.sqrt(1000*i)
        
        

    d1_numerator = np.log(current_price/strike_price) + (risk_free + sigma**2/2) * (T - current_time)
    d1_denominator = sigma * np.sqrt(T - current_time)

    d1 = d1_numerator / d1_denominator
    d2 =  d1 - d1_denominator

    if type == "call":
        analytic_price = current_price*norm.cdf(d1) - (norm.cdf(d2)*strike_price*np.exp(-risk_free * (T - current_time)))

    else:
        analytic_price = -current_price*norm.cdf(-d1) + (norm.cdf(-d2)*strike_price*np.exp(-risk_free * (T - current_time)))


    print(" ", end="\n\n")
    print(f"Analytical European {type} option value: {analytic_price}")
    print(f"Monte carlo European {type} option value: {mput_estimates[numb-1]}")

    # 4. Plot the Monte Carlo estimates, the analytical European put option value, 
    # and three standard deviation error bounds.
    if plot:
        plt.figure(figsize = [11, 6])
        plt.plot([analytic_putprice]*numb)
        plt.plot(mput_estimates, ".")
        plt.plot(analytic_putprice + np.array(mput_std)*3)
        plt.plot(analytic_putprice - np.array(mput_std)*3)
        plt.fill_between(np.arange(num), analytic_putprice + np.array(mput_std)*3, 
                        analytic_putprice - np.array(mput_std)*3,facecolor='whitesmoke', interpolate=True)
        plt.ylabel(f"{type.capitalize()} option prices")
        plt.xlabel("Number of simulations")
        plt.title(f"Monte carlo estimation of European {type.capitalize()} option value")
        plt.show()




option_prices(current_price, risk_free, sigma, T, current_time, type="call", plot=True)

#Share information
sigma = 0.3
r = 0.1
S0 = 100

#Option information
T = 0.5
K = 110

#Function for terminal share valuation
def terminal_shareprice2(S_0, risk_free_rate,sigma,Z,T):
    """Generates the terminal share price given some random normal values, Z"""
    return S_0*np.exp((risk_free_rate-sigma**2/2)*T+sigma*np.sqrt(T)*Z)

#Function for put valuations
def put_price(S_0,K,risk_free_rate,sigma,Z,T):
    """Function for evaluating the call price in Monte Carlo Estimation"""
    share_terminal = terminal_shareprice2(S_0, risk_free_rate, sigma, Z, T)
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
