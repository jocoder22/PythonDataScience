#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as opt

# parametes for option
r = 0.06
s0 = 100
v0 = 0.06
k = np.array([110,100,90])
price = [8.02,12.63,18.72]
T = 1
k_log = np.log(k)
k_log.shape = (3,1)
rho = -0.4

# parameter for Gil-Paelez
t_max = 30
N = 100

# define characteristic function
def a(sigma):
  return sigma**2/2

def b(u,theta,kappa,sigma):
  return kappa - rho*sigma*1j*u

def c(u, theta, kappa,sigma):
  return -(u**2+1j*u)/2

def d(u, theta, kappa, sigma):
  return np.sqrt(b(u,theta,kappa,sigma)**2 -4*a(sigma)*c(u, theta, kappa, sigma))

def xminus(u, theta, kappa, sigma):
  return (b(u,theta,kappa,sigma)-d(u,theta,kappa, sigma))/(2*a(sigma))

def xplus(u,theta,kappa, sigma):
  return (b(u,theta,kappa,sigma)+d(u,theta,kappa, sigma))/(2*a(sigma))

def g(u,theta, kappa, sigma):
  return xminus(u,theta,kappa, sigma)/xplus(u,theta,kappa,sigma)

def C(u, theta, kappa, sigma):
  val1 = T*xminus(u, theta, kappa, sigma)
  val12 = np.log((1-g(u,theta, kappa, sigma)*np.exp(-T*d(u,theta,kappa,sigma)))
                 /(1-g(u,theta,kappa, sigma)))/a(sigma)
  val0 = val1 - val12
  return r*T*1j*u + theta*kappa*val0

def D(u, theta, kappa, sigma):
  va1 = 1-np.exp(-T*d(u, theta, kappa, sigma))
  va2 = 1-g(u, theta, kappa, sigma) * np.exp(-T*d(u,theta, kappa, sigma))
  return (va1/va2)*xminus(u,theta, kappa, sigma)

def log_char(u, theta, kappa, sigma):
  return np.exp(C(u,theta, kappa, sigma) + D(u,theta, kappa, sigma)*v0 + ij*u*np.log(s0))

def adj_char(u,theta, kappa, sigma):
  return log_char(u-1j, theta, kappa, sigma)/log_char(-ij, theta, kappa, sigma)


# variables for Gil-Pelaez
delta_t = t_max/N
from_1_to_N = np.linspace(1,N,N)
t_n = (from_1_to_N - 1/2)*delta_t

def Hest_Pricer(x):
  theta = x[0]
  kappa = x[1]
  sigma = x[2]
  
  first_integral = np.sum((((np.exp(-1j*t_n*k_log)*adj_char(t_n,theta,kappa,sigma)).imag)/t_n)*delta_t, axis=1)
  second_integral = np.sum((((np.exp(-1j*t_n*k_log)*log_char(t_n,theta,kappa,sigma)).imag)/t_n)*delta_t, axis=1)
  
  fourier_callval = s0*(1/2 + first_integral/np.pi) - np.exp(-r*T)*k*(1/2 + second_integral/np.pi)
  
  
def opt_func(x):
  return sum(np.abs(price - Hest_Pricer(x)))


# calibrating our model
opt_vals = opt.fmin_slsqp(opt_func, (0.1,3,0.1))

theta_hat = opt_vals[0]  # 0.05988378
kappa_hat = opt_vals[1]   # 3.07130476
sigma_hat = opt_vals[2]   # 0.25690418

