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
x= 40           # Strike/Exercise price
T= 0.5          # Maturity (in years)
r= 0.05         # Risk-free rate
sigma= 0.2      # Annualized volatility



n_simulation= 100 # Number of simulations and steps
n_steps= 100
dt= T/n_steps

call= np.zeros([n_simulation], dtype=float)

for j in range(0,n_simulation):
    sT=s0
    total=0
    for i in range(0,int(n_steps)):
        e = norm.rvs()
        sT*= np.exp((r-0.5*sigma**2)*dt+sigma*e*np.sqrt(dt))
        total+=sT
    price_average = total/n_steps
    call[j]=max(price_average-x, 0)

call_price=np.mean(call)*np.exp(-r*T)

print(f'Call price = {round(call_price, 3)}')



