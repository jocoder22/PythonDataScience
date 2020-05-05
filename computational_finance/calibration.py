import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as opt
from functools import partial

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

def optionprice(sigma, type="call"):
  if type == "call":
    return norm.cdf(d1(sigma))*s0 - norm.cdf(d2(sigma))*Kc*np.exp(-r * Tc)
  else:
    return -norm.cdf(-d1(sigma))*s0 + norm.cdf(-d2(sigma))*Kc*np.exp(-r * Tc)

def F(sigma, type="call"): # this gives F(x) = 0
  return callprice(sigma, type="call") - price

# finding sigma
sigma_val = opt.broyden1(F, 0.2, args=("call")) # 0.2 is a random initializaton value

print(sigma_val)


def optionprice(sigma, type="call"):
  if type == "call":
    return norm.cdf(d1(sigma))*s0 - norm.cdf(d2(sigma))*Kc*np.exp(-r * Tc)
  else:
    return -norm.cdf(-d1(sigma))*s0 + norm.cdf(-d2(sigma))*Kc*np.exp(-r * Tc)



def F(sigma, type="call"): # this gives F(x) = 0
  return optionprice(sigma, type="call") - price


G_partial = partial(F, type="call")

# finding sigma
sigma_val2 = opt.broyden1(G_partial, 0.2) # 0.2 is a random initializaton value

print(sigma_val2)
# Find the put price

def F3(sigma, typef="call"): # this gives F(x) = 0
  return optionprice(sigma, type=typef) - price

sigma_val3 = opt.broyden1(F3, xin = 0.1) 
print(sigma_val3)


# calculate the put price
put_price = optionprice(sigma_val3, type="put")
print(put_price)
  
