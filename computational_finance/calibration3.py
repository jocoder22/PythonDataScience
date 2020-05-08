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
  return sigma**/2

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
