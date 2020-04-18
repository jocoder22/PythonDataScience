#!/usr/bin/env python
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



def anal_option_prices(current_price, risk_free, sigma, term, strike_price, current_time=0, type="call"):
    """The Ana_option_prices function calculate both analytical price for either call or put option
 
    Args: 
        present_price (float/int): initial share price
        riskfree (float/int): risk free rate
        sigma (float/int): share volatility
        Z (float/int): normal random variables
        T (float/int): term of share price
        type (str): type of option, "call" or "put"
        strike_price (float/int) : strike pricke
 
    Returns: 
        price (float/int): price of the option
 
    """   
    
    # calculate d1 and d2
    d1_numerator = np.log(current_price/strike_price) + (risk_free + sigma**2/2) * (T - current_time)
    d1_denominator = sigma * np.sqrt(T - current_time)

    d1 = d1_numerator / d1_denominator
    d2 =  d1 - d1_denominator


    if type == "call":
        analytic_price = current_price*norm.cdf(d1) - (norm.cdf(d2)*strike_price*np.exp(-risk_free * (T - current_time)))

    else:
        analytic_price = -current_price*norm.cdf(-d1) + (norm.cdf(-d2)*strike_price*np.exp(-risk_free * (T - current_time)))

        
    return analytic_price



# define characteristic functions
def c_M1_t(s0, r, sigma, T, t):
  """
  
  
  """
  
  s_ij = 1j*t*(np.log(s0) + (r - sigma**2/2)*T)
  
  sigma_t = (sigma**2)* T * (t**2/2)
  
  return np.exp(s_ij - sigma_t)


def c_M2_t(s0, r, sigma, T, t):
  """
  
  
  """
  
  sigma_ij = np.exp(1j*t*sigma**2*T)
  
  M1_t = c_M1_t(s0, r, sigma, T, t)
  
  return sigma_ij*M1_t



def fourier_option_prices(s0, r, sigma, T, K):
  """
  
  
  
  """
  # technique for approximating integral value (using areas of rectangles)
  t_max = 20
  N = 100
  k_log = np.log(K)

  # calculating delta 
  delta_t = t_max/N
  t_range = np.linspace(1,N, N)
  t_n = (t_range - 1/2)* delta_t

  cm1t = c_M2_t(s0, r, sigma, T, t_n)
  cm2t = c_M1_t(s0, r, sigma, T, t_n)

  s0_integral = sum((((np.exp(-1j*t_n*k_log)*cm1t).imag)/t_n)*delta_t)
  k_integral = sum((((np.exp(-1j*t_n*k_log)*cm2t).imag)/t_n)*delta_t)  
  
  s0_part = s0*(1/2 + s0_integral/np.pi)
  k_part = np.exp(-r*T)*K*(1/2 + k_integral/np.pi)
  
  return s0_part - k_part


def get_npi(b2, b1, d, c, n):
  """
  
  
  """

  npi_d = np.pi*n*(d-b1)/(b2-b1)
  npi_c = np.pi*n*(c-b1)/(b2-b1)
  npi_2 = np.pi*n/(b2-b1)
  
  return npi_d, npi_2, npi_c

  
def upsilon_n(b2, b1, d, c, n):
  """
  
  
  """
  
  a, b, cc = get_npi(b2, b1, d, c, n)
  
  val_one = (np.cos(a)*np.exp(d) - np.cos(cc)*np.exp(c))
  val_two = (b*(np.sin(a)*np.exp(d) - np.sin(cc)*np.exp(c)))
  
  return (val_one + val_two) / (1 + (b**2))
  
  
def psi_n(b2, b1, d, c, n):

  """
  
  
  """
  a, b, cc = get_npi(b2, b1, d, c, n) 
  
  if n == 0:
    return d - c
  
  else:
    return 1/b * (np.sin(a) - np.sin(cc))
  
  

def v_n(K, b2, b1, n):
  """
  
  
  """
  
  
  up_silon = upsilon_n(b2, b1, b2, 0, n)
  
  psi_ = psi_n(b2, b1, b2,  0, n)
  
  return 2*K*(up_silon - psi_)/(b2 - b1)


def logchar_func(u, s0, r, sigma, K, T):
  """
  
  
  """
  
  s_ij1 = 1j*u*(np.log(s0/K) + (r - sigma**2/2)*T)
              
  sigma_t1 = (sigma**2) * T * (u**2)/2
  
  return np.exp(s_ij1 - sigma_t1)




def call_price(NN, s0, sigma, r, K, T, b2, b1):
  """
  
  
  """
  
  vn = v_n(K, b2, b1, 0)
  log_char = logchar_func(0, s0, r, sigma, K, T)
  
  price = vn * log_char / 2
  
  for n in range(1,NN):
    b = np.pi*n/(b2-b1)
    vnn = v_n(K, b2, b1, n)
    log_charn = logchar_func(b, s0, r, sigma, K, T)
    exp_n = np.exp(-1j*b*b1)
    
    price = price +  log_charn * exp_n * vnn
  
  
  return price.real*np.exp(-r * T)


# share specific information
s0 = 100
r = 0.06
sigma = 0.3


# option specific information
K = 110
T = 1
k_log = np.log(K)


analytic_callprice = anal_option_prices(s0, r, sigma, T, K, type="call")
fourier_call = fourier_option_prices(s0, r, sigma, T, K)

print(analytic_callprice, fourier_call)


# b1, b2 for call
c1 = r
c2 = T*sigma**2
c4 = 0
L = 10

b1 = c1 - L * np.sqrt(c2 - np.sqrt(c4))
b2 = c1 + L * np.sqrt(c2 - np.sqrt(c4))


# calculate COS for various N
COS_callprice = np.zeros(50)


for i in range(1,51):
  COS_callprice[i-1] = call_price(i, s0, sigma, r, K, T, b2, b1)
  
print(COS_callprice)
# plotting the results
plt.plot(COS_callprice)
plt.plot([analytic_callprice]*50)
plt.xlabel("N")
plt.ylabel("Call price")
plt.show()


# plot the log absolute error
plt.plot(np.log(np.absolute(COS_callprice - analytic_callprice)))
plt.xlabel("N")
plt.ylabel("Log absolute error")
plt.show()




 




