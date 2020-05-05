import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as opt


# Set general parameters
s0 = 100
r = 0.1  # from govt bond prices and their yields

# set put parameters
# sigma and price are unknown
Kp = 110
Tp = 2

# set call parameters
Kc = 95
Tc = 1
price = 15

# define functions
def d1(sigma):
  return 1/(sigma * np.sqrt(Tc)) * (np.log(s0/Kc) + (r+ sigma**2/2)*Tc)

def d2(sigma):
  return d1(sigma) - sigma*np.sqrt(Tc)

def callprice(sigma):
  return norm.cdf(d1(sigma))*s0 - norm.cdf(d2(sigma))*Kc*np.exp(-r * Tc)

def F(sigma):
  return callprice(sigma) - price

# finding sigma
sigma_val = opt.broyden1(F, 0.2) # 0.2 is a random initializaton value

print(sigma_val)
