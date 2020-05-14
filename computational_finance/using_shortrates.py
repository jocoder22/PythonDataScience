#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from short_rates import vasicek_mean, vasicek_var

# set market parameters
r0 =  0.05
alpha = 0.2
b = 0.08
sigma = 0.025

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

z_month1 = norm.rvs(size = [nsimulations, nyears])
z_month2 = norm.rvs(size = [nsimulations, nyears])

r_simtemp = np.zeros([nsimulations, nyears+1])
y_simtemp = np.zeros([nsimulations, nyears+1])


r_simtemp[:,0] = r0
y_simtemp[:,0] = y0


correlations = ry_rho(t[0:-1],t[1:])

# generate correlated standard normals
z_month2 = correlations*z_moth1 + np.sqrt(1-correlations**2)*z_month2

for i in range(nyears):
  sim_r_mean = vasicek_mean(r_simtemp[:,i],t[i],t[t+1])
  sim_r_val = np.sqrt(vasicek_var(t[i],t[i+1]))*z_moth1[:,i]
  r_simtemp[:,0] = sim_r_mean + sim_r_val
  
  sim_y_mean = y_mean(y_simtemp[:,i],r_simtemp[:,i], t[i],t[t+1])
  sim_y_val = np.sqrt(y_var(t[i],t[i+1]))*z_moth2[:,i] 
  y_simtemp[:,0] = sim_y_mean + sim_y_val

zcb_prices = np.mean(np.exp(-y_simtemp), axis=0)

# Yt estimates
r_mat = np.cumsum(r_simtem[:,0:-1], axis = 1) * (t[1:] - t[0:-1])
r_mat2 = np.cumsum(r_simtem[:,0:-1] + r_simtem[:,1:], axis = 1)/2 * (t[1:] - t[0:-1])


# bond prices estimates
squad_prices = np.ones(nyears+1)
trap_prices = np.ones(nyears+1)

squad_prices[1:] = np.mean(np.exp(-r_mat), axis = 0)
trap_prices[1:] = np.mean(np.exp(-r_mat2), axis = 0)

# calculate closed form bond prices
bond_vec = bond_price(r0,0,t)

# plotting bond prices
plt.plot(t, bond_vec, label="Analytical solution: Closed-form")
plt.plot(t, zcb_prices, ".", label="Joint Simulation: Simulated Yt and rt")
plt.plot(t, squad_prices, "x", label="Simple Quadrature: Simulated rt and estimated Yt")
plt.plot(t, trap_prices, "^", label="Trapezoidal Quadrature: Simulated rt and estimated Yt")
plt.legend()
plt.show()

# calculate bond yields
bond_yield = -np.log(bond_vec[1:])/t[1:])
month_yield = -np.log(zcb_prices[1:])/t[1:])
squad_yield = -np.log(squad_prices[1:])/t[1:])
trap_yield = -np.log(trap_prices[1:])/t[1:])

# plotting bond yields
plt.plot(t[1:], bond_yield*100, label="Analytical solution: Closed-form")
plt.plot(t[1:], month_yield*100, ".", label="Joint Simulation: Simulated Yt and rt")
plt.plot(t[1:], squad_yield*100, "x", label="Simple Quadrature: Simulated rt and estimated Yt")
plt.plot(t[1:], trap_yield*100, "^", label="Trapezoidal Quadrature: Simulated rt and estimated Yt")
plt.legend()
plt.show()



