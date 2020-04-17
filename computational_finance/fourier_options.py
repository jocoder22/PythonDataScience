
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


# share specific information
s0 = 100
r = 0.06
sigma = 0.3


# option specific information
K = 110
T = 1
k_log = np.log(K)

