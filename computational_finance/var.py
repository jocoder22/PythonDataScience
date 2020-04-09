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


sp = {"end": "\n\n", "sep":"\n\n"}

#General share information
#Share prices. Portfolio contains 3 shares 
S0 = np.array([[100],[95],[50]])

#Share sigmas (the volatility of each share) 
sigma = np.array([[0.15],[0.2],[0.3]])

#Correlation Matrix 
cor_mat = np.array([[1,0.2, 0.4],[0.2,1,0.8],[0.4,0.8,1]])
L = np.linalg.cholesky(cor_mat) #Cholesky decomposition
L
np.shape(L)

#Risk-free interest rate
r = 0.1
T = 1


#Applying Monte Carlo estimation of VaR
np.random.seed(0)
t_simulations = 10000

alpha = 0.05
#Current portfolio value

#Current portfolio value = stock1 + stock2 + stock3 
portval_current = np.sum(S0)
portval_current
np.shape(portval_current)

#Terminal share function
#Simulate stock price using Geometric Brownian Motion. 
def terminal_shareprice(S_0, risk_free_rate,sigma,Z,T):
    """Generates the terminal share price given some random normal values, Z"""
    return S_0*np.exp((risk_free_rate-sigma**2/2)*T+sigma*np.sqrt(T)*Z)

#Creating 10000 simulations of future portfolio values
#Drawing random numbers from a normal distribution 
Z = np.matmul(L,norm.rvs(size = [3,t_simulations]))
Z
np.shape(Z) 

#Use sum function as we add the simulations for each of the three stock evolution to obtain portfolio value evolution 
portval_future = np.sum(terminal_shareprice(S0,r,sigma,Z,T),axis = 0)
portval_future
np.shape(portval_future)
plt.plot(portval_future)


#Calculating portfolio returns
portreturn = (portval_future - portval_current)/portval_current
portreturn


#Sorting returns
portreturn = np.sort(portreturn)
portreturn

#Determining VaR
mVaR_estimate = -portreturn[int(np.floor(alpha*t_simulations))-1]
mVaR_estimate
print(portreturn, mVaR_estimate, **sp)



###########################################################################################################
############################ REAL DATA ####################################################################
###########################################################################################################

stocksname = ['AAPL', 'MSFT', 'GOOGL', 'NFLX', 'TSLA', 'AMZN', 'TM', 'JPM', 'C']

startdate = datetime(2016, 4, 15)
enddate = datetime(2018, 4, 10)

stock = pdr.get_data_yahoo(stocksname, startdate, enddate)[['Adj Close']]

print(stock.tail(), stock.iloc[:,-1], *sp)