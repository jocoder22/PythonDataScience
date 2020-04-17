
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





def fourier_option_prices(s0):
  """
  
  
  
  """
  # technique for approximating integral value (using areas of rectangles)
  t_max = 20
  N = 100

  # calculating delta 
  delta_t = t_max/N
  t_range = np.linspace(1,N, N)
  t_n = t_range * delta_t


  s0_integral = sum(((np.exp(-1j*t_n*k_log)*(c_M2_t(t_n)).imag)/t_n)*delta_t)
  k_integral = sum(((np.exp(-1j*t_n*k_log)*(c_M1_t(t_n)).imag)/t_n)*delta_t)  
  
  s0_part = s0*(1/2 + s0_integral/np.pi)
  k_part = np.exp(-r*T)*K*(1/2 + k_integral/np.pi)
  
  return s0_part - K_part


def get_npi(b2, b1, c, n):
  """
  
  
  """
  npi_d = np.pi*n*(d-b1)/(b2-b1)
  npi_c = np.pi*n*(c-b1)/(b2-b1)
  npi_2 = npi.pi*n/(b2-b1)
  
  return npi_d, npi_2, npi_c

  
def upsilon_n(b2, b1, c, n):
  """
  
  
  """
  
  a, b, c = get_npi(b2, b1, c, n)
  
  val_one = (np.cos(a)*np.exp(d) - np.cos(c)*np.exp(c))
  val_two = (b*(np.sin(a)*np.exp(d) - np.sin(c)*np.exp(c)))
  
  return (val_one + val_two) / (1 + b**2)
  
  
 def psi_n(b2, b1, d, c, n):
  """
  
  
  """
  a, b, c = get_npi(b2, b1, c, n)
  
  if n == 0:
    return d - c
  
  else:
    return (1/b * (np.sin(a) - np.sin(c)))
  
  
def v_n(K, b2, b1, n):
  """
  
  
  """
  
  
  up_silon = upsilon_n(b2, b1, 0, n)
  
  psi_ = psi_n(b2, b1, 0, n)
  
  return 2*K*(up_silon - psi_)


def logchar_func(u, s0, r, sigma, K, T):
  """
  
  
  """
  
  s_ij1 = 1j*u*(np.log(s0/K) + (r - sigma**2/2)*T)
              
  sigma_t1 = sigma**2/2 * T * u**2/2
  
  return np.exp(s_ij1 - sigma_t1)


def call_price(N, s0, sigma, r, K, T, b2, b1):
  """
  
  
  """
  
  a, b, c = get_npi(b2, b1, c, n)
  
  vn = v_n(K, b2, b1, 0)
  log_char = logchar_func(0, s0, r, sigma, K, T)
  
  price = vn * (log_char / 2)
  
  for n in range(1,N):
    vnn = v_n(K, b2, b1, n)
    log_charn = logchar_func(a, s0, r, sigma, K, T)
    exp_n = (-ij*b*b1)
    
    price += logcharn * exp_n * vnn
  
  
  return price.real*np.exp(-r * T)


# b1, b2 for call
c1 = r
c2 = T*sigma**2
c4 = 0
L = 10

b1 = c1 - L * np.sqrt(c2-np.sqrt(c4))
b2 = c1 + L * np.sqrt(c2-np.sqrt(c4))



 




