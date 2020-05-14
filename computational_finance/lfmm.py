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

nyears = 10
simulations = 10

t = np.array(range(0,nyears+1))

z = norm.rvs(size = [simulations, nyears])
r_sim = np.zeros([simulations, nyears])
r_sim[:,0] = r0
vasicek_mean_vector = np.zeros(nyears+1)


for i in range(nyears):
  r_sim[:,i+1] = vasicek_mean(r_sim[:,i],t[i], t[i+1]) + np.sqrt(vasicek_var(t[i], t[i+t])) * z[:,i]
  
  
s_mean = r0 * np.exp(-alpha*t) + b*(1-np.exp(-alpha*t))
