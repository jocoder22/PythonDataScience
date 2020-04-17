
# We will be pricing a vanilla European call option on a single stock under the following conditions:
# o Continuously-compounded interest rate, 𝑟, of 6%
# o Initial stock price, 𝑆0, of $100
# o Stock volatility, 𝜎, 30%
# o Strike price, 𝐾, of $110
# o Maturity time, 𝑇, of one year
# As per usual, we make all the assumptions of the Black-Scholes model.


# import necessary modules
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from option_pricing import anal_option_prices


# share specific information
s0 = 100
r = 0.06
sigma = 0.3


# option specific information
K = 110
T = 1
k_log = np.log(K)


anlytic_callprice = anal_option_prices(s0, r, sigma, T, type="call")


# define characteristic functions
def c_M1_t(t):
  """
  
  
  """
  
  s_ij = 1j*t*(np.log(s0) + (r - sigma**2/2)*T)
  sigma_t = sigma**2/2 * T * t**2/2
  
  return np.exp(s_ij - sigma_t)


def c_M2_t(t):
  """
  
  
  """
  
  sigma_ij = np.exp(1j*t*sigma**2/2*T)
  
  M1_t = c_M1_t(t)
  
  return sigma_ij*M1_t






               
