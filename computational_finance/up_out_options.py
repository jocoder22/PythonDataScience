#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


np.random.seed(112)


s0= 40          # Today's stock price
K= 40           # Strike/Exercise price
T= 0.5          # Maturity (in years)
r= 0.05         # Risk-free rate
sigma= 0.2      # Annualized volatility
barrier = 42    # Barrier level


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
    # np.random.seed(112)

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

    # print(f'Call price = {round(callprice, 3)}')

    return callprice

black_schole_callprice(s0,K,T,r,sigma)


def up_and_out_call(s0,K,T,r,sigma,barrier):
    """
        Returns: Call value of an up-and-out barrier option with European call
    """

    n_simulation= 100 # Number of simulations and steps
    n_steps= 100 # Define number of steps.
    dt = T/n_steps
    total=0

    # simulates option price
    for j in range(0,n_simulation):
        sT=s0
        out=False

        # simulate the stock price evolution
        for i in range(0,int(n_steps)):
            e = norm.rvs()
            sT *= np.exp((r-sigma**2/2)*dt+sigma*e*np.sqrt(dt))

            # out whenever the stock price exceeds the barrier
            if sT > barrier:
                out=True
        if out==False:
            total += black_schole_callprice(s0,K,T,r,sigma)

    return total/n_simulation
    


s0= 40              # Stock price today
K= 40               # Strike price
barrier = 42        # Barrier level
T= 0.5              # Maturity in years
r=0.05              # Risk-free rate
sigma=0.2           # Annualized volatility
n_simulation = 100  # number of simulations

result = up_and_out_call(s0,K,T,r,sigma,barrier)
print('Price for the Up-and-out Call = ', round(result,3))

