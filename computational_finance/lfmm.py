#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random

from short_rates import vasicek_mean, vasicek_var

# set market parameters
r0 =  0.05
alpha = 0.2
b = 0.08
sigma = 0.025


# problem parameters
t =  np.linspace(0,40,21)
sigmaj = 0.2


#  define useful functions
# Analytical bond price (ZCB)
def a_part(t1,t2):
  """
  
  """
  numer_ = 1 - np.exp(-alpha * (t2-t1))
  
  result = numer_/alpha
  
  return result



def d_part(t1, t2):
  """
  
  
  
  """
  
  val2 =  (t2 - t1 - a_part(t1,t2))*  (sigma**2/(4*alpha**2)-b)
  val3 = sigma**2/2 * a_part(t1,t2)**2 / (4*alpha)
  
  result = val2 - val3
  
  return result
  
def bond_price(r,t,T):
  """
  
  
  """
  bondprice = np.exp(-a_part(t,T)*r*d_part(t,T))
  
  return bondprice

vasicek_bond - bond_price(r0,0,t)

# simulate interest rate paths
np.random.seed(0)

nyears = len(t)
simulations = 100000

mc_forward = np.ones([simulations, nyears-1])*(vasicek_bond[:-1] - vasicek_bond[1:])/(2*vasicek_bond[1:])
predcorr_forward = np.ones([simulations, nyears-1])*(vasicek_bond[:-1] - vasicek_bond[1:])/(2*vasicek_bond[1:])
predcorr_capfac = np.ones([simulations, nyears])
mc_capfac = np.ones([simulations, nyears])

delta = np.ones([simulations, nyears-1])*(t[1:] - t[:-1])
z = norm.rvs(size = [simulations, 1])


for i in range(1, nyears):
  z = norm.rvs(size = [simulations, 1])
  
  # explicit Monte Carlo simulation
  muhat = np.cumsum(delta[:,i:]*mc_forward[:,i:]*sigmaj**2/(1 + delta[:,i:]*mc_forward[:,i:]), axis =1)
  mc_forward[:,i:] = mc_forward[:,i:]*np.exp((muhat - sigmaj**2/2)*delta[:,i:]+sigmaj*np.sqrt(delta[:,i:])*z)
  
  # Predictor-Corrector Monte Carlo simulation
  mu_ = np.cumsum(delta[:,i:]*predcorr_forward[:,i:]*sigmaj**2/(1 + delta[:,i:]*predcorr_forward[:,i:]), axis =1)
  for_temp = predcorr_forward[:,i:]*np.exp((mu_ - sigmaj**2/2)*delta[:,i:]+sigmaj*np.sqrt(delta[:,i:])*z)
  mu_temp = np.cumsum(delta[:,i:]*for_temp[:,i:]*sigmaj**2/(1 + delta[:,i:]*for_temp[:,i:]), axis =1)
  predcorr_forward[:,i:] = predcorr_forward[:,i:]*np.exp((mu_  + mu_temp - sigmaj**2/2)*delta[:,i:]+sigmaj*np.sqrt(delta[:,i:])*z)
  
  
# Implying capitalisation factors from the forward rates 
mc_capfac[:,i:] = np.cumprod(1 + delta*mc_forward, axis =1) 
predcorr_capfac[:,i:] = np.cumprod(1 + delta*predcorr_forward, axis =1) 


# Inverting the capitalisation factors to imply bond prices (discount factors)
mc_prices = mc_capfact**(-1)
predcorr_prices = predcorr_capfac**(-1)

# Taking averages
mc_final = np.mean(mc_prices, axis=0)
predcorr_final = np.mean(predcorr_prices, axis=0)
