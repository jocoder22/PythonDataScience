#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from short_rates import vasicek_mean, vasicek_var

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
  
  val1 = b - (sigma**2/(2*alpha**2))
  val2 = a_part(t1,t2) - (t2 - t1)
  val3 = sigma**2 * a_part(t1,t2)**2 / (4*alpha)
  
  result = val1 * val2 - val3
  
  return result
  
def bond_price(r,t,T):
  """
  
  
  """
  bondprice = np.exp(-a_part(t,T)*r*d_part(t,T))
  
  return bondprice


def y_mean(y,r,t1,t2):
  """
  
  """
  ymean = y + (t2-t1)*b + (r-b)*a_part(t1,t2)
  
  return ymean


def y_var(t1,t2):
  """
  
  """
  var1 = sigma**2/alpha**2
  val2 = t2-t1 - a_part(t1,t2) 
  val3 = alpha*(a_part(t1,t2)**2/2)
  
  yvar = var1 * (val2 - val3)
  
  return yvar

def ry_var(t1,t2):
  """
  
  
  """
  ry_covariance = sigma**2*(a_part(t1,t2)**2)/2
  
  return ry_variance

def ry_rho(t1,t2):
  """
  
  
  """
  ry_variance = ry_var(t1,t2)
  ry_stds = np.sqrt(vasicek_var(t1,t2)*y_var(t1,t2))
  ry_correlation = ry_variance / ry_stds
  
  return ry_correlation
    
  
# initialize y value
y0 = 0
np.random.seed(0)

# initialize number of years and number of simulations
nyears = 10
nsimulations = 100000

t = np.array(range(0, nyears+1))

z_mont1 = norm.rvs(size = [nsimulations, nyears])
z_mont2 = norm.rvs(size = [nsimulations, nyears])

r_simtemp = np.zeros([nsimulations, nyears+1])
y_simtemp = np.zeros([nsimulations, nyears+1])


r_simtemp[:,0] = r0
y_simtemp[:,0] = y0




