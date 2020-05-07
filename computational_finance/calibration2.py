#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as opt


r0 = 0.05
# create yield curve for integer maturities
years = np.linspace(1,10,10)

# We suppose the yield curve for maturities from 1 to 10 yrs takes the form
# y(t) = 1/75 * (t)**1/5 + 0.04
yield_curve = (years)**(1/5)/75 + 0.04
bond_prices = np.exp(-yield_curve*year)


# plot yield curve
plt.plot(years, yield_curve*100)
plt.xlabel("Maturity")
plt.ylabel("Yield")
plt.title("Yield Curve")
plt.show


# calibrating Vasicek model
# Analytical bond price
def A_part(t1, t2, alpha):
  return (1- np.exp(-alpha*(t2-t1)))/alpha

def D_part(t1, t2, alpha, b , sigma):
  val1 = (t2-t1 -A_part(ti, t2, alpha))*(sigma**2/2 / (2*alpha**2)-b)
  val2 = sigma**2/2*A_part(t1,t2,alpha)**2/(4*alpha)
  return val1 - val2

def bondprice(r, t, T, alpha,b,sigma):
  return np.exp(-A_part(t,T,alpha)*r + D_part(t,T,alpha, b,sigma))

