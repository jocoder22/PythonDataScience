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

#General share information
#Share prices. Portfolio contains 3 shares 
S0 = np.array([[100],[95],[50]])

#Share sigmas (the volatility of each share) 
sigma = np.array([[0.15],[0.2],[0.3]])

#Correlation Matrix 
cor_mat = np.array([[1,0.2, 0.4],[0.2,1,0.8],[0.4,0.8,1]])
L = np.linalg.cholesky(cor_mat) #Cholesky decomposition
print2(np.shape(L), S0.shape)

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
print2(np.shape(portval_current), portval_current)

#Terminal share function
#Simulate stock price using Geometric Brownian Motion. 
def terminal_shareprice(S_0, risk_free_rate,sigma,Z,T):
    """Generates the terminal share price given some random normal values, Z"""
    return S_0*np.exp((risk_free_rate-sigma**2/2)*T+sigma*np.sqrt(T)*Z)

#Creating 10000 simulations of future portfolio values
#Drawing random numbers from a normal distribution 
Z = np.matmul(L,norm.rvs(size = [3,t_simulations]))

#Use sum function as we add the simulations for each of the three stock evolution to obtain portfolio value evolution 
portval_future = np.sum(terminal_shareprice(S0,r,sigma,Z,T),axis = 0)
np.shape(portval_future)
plt.plot(portval_future)
plt.show()


#Calculating portfolio returns
portreturn = (portval_future - portval_current)/portval_current


#Sorting returns
portreturn = np.sort(portreturn)

#Determining VaR
mVaR_estimate = -portreturn[int(np.floor(alpha*t_simulations))-1]
print2(portreturn, mVaR_estimate)



###########################################################################################################
############################ REAL DATA ####################################################################
###########################################################################################################

stocksname = ['AAPL', 'MSFT', 'GOOGL', 'NFLX', 'TSLA', 'AMZN', 'TM', 'JPM', 'C']

# startdate = datetime(2017, 4, 15)
# enddate = datetime(2018, 4, 10)
# startdate = datetime(2017, 4, 15)
# stock = pdr.get_data_yahoo(stocksname, startdate, enddate)[['Adj Close']]


startdate = datetime(2019, 4, 15)
stock = pdr.get_data_yahoo(stocksname, startdate)[['Adj Close']]

#General share information
#Share prices. Portfolio contains 9 shares 
S0 = np.array(stock.iloc[-1,:]).reshape(stock.shape[1], -1)


#Current portfolio value = stock1 + stock2 + stock3 ...... + stock9
portval_current = np.sum(S0)
print2(np.shape(portval_current), portval_current)


#Share sigmas (the volatility of each share) 
sigma = np.array(stock.std()).reshape(stock.shape[1], -1)

#Correlation Matrix 
cor_mat = np.array(stock.corr())
L = np.linalg.cholesky(cor_mat) #Cholesky decomposition

# print2(S0, sigma, cor_mat, L)
# print2(stock.tail(), stock.iloc[-1,:], stock.std(), np.array(stock.corr()))

#Creating 10000 simulations of future portfolio values
#Drawing random numbers from a normal distribution 
Z = np.matmul(L,norm.rvs(size = [stock.shape[1], t_simulations]))



#Use sum function as we add the simulations for each of the nine stock evolution to obtain portfolio value evolution 
portval_future = np.sum(terminal_shareprice(S0,r,sigma,Z,T),axis = 0)
np.shape(portval_future)
plt.plot(portval_future)
plt.show()

#Calculating portfolio returns
portreturn = (portval_future - portval_current)/portval_current


#Sorting returns
portreturn = np.sort(portreturn)

#Determining VaR
mVaR_estimate = -portreturn[int(np.floor(alpha*t_simulations))-1]
print(portreturn, mVaR_estimate, **sp)

