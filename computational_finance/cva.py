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