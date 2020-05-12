#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as opt
import matplotlib.pyplot as plt

r0 = 0.05
# create yield curve for integer maturities
years = np.linspace(1,10,10)

# We suppose the yield curve for maturities from 1 to 10 yrs takes the form
# y(t) = 1/75 * (t)**1/5 + 0.04
yield_curve = (years)**(1/5)/75 + 0.04
bond_prices = np.exp(-yield_curve*years)

# plot yield curve
plt.plot(years, yield_curve*100)
plt.xlabel("Maturity")
plt.ylabel("Yield")
plt.title("Yield Curve")
plt.show()


# calibrating Vasicek model
# Analytical bond price
def A_part(t1, t2, alpha):
  return (1- np.exp(-alpha*(t2-t1)))/alpha

def D_part(t1, t2, alpha, b , sigma):
  val1 = (t2-t1 -A_part(t1, t2, alpha))*(sigma**2/(2*alpha**2)-b)
  val2 = sigma**2*A_part(t1,t2,alpha)**2/(4*alpha)
  return val1 - val2

def bondprice(r, t, T, alpha,b,sigma):
  return np.exp(-A_part(t,T,alpha)*r + D_part(t,T,alpha, b,sigma))

# the F function
# find the values for which the differences between the bondprice and yield curve bond prices are minimized
def F(x):
  alpha = x[0]
  b = x[1]
  sigma = x[2]
  return sum(np.abs(bondprice(r0,0,years,alpha,b,sigma) - bond_prices))



# minimizing F function
bnds = ((0,1),(0,0.2), (0,0.2))
opt_value = opt.fmin_slsqp(F, (0.3,0.05,0.03), bounds=bnds)
opt_alpha = opt_value[0]
opt_b = opt_value[1]
opt_sigma = opt_value[2]


# Calculating model prices and yield
model_prices = bondprice(r0,0,years, opt_alpha, opt_b, opt_sigma)
model_yield =  -np.log(model_prices)/years

# plotting prices
plt.plot(years, bond_prices, label="Market prices")
plt.plot(years, model_prices, ".", label ="Calibarted prices")
plt.xlabel("Maturity")
plt.ylabel("Bond price")
plt.legend()
plt.show()

# plotting yields
plt.plot(years, yield_curve*100, label="Yield curve")
plt.plot(years, model_yield*100, "x", label ="Calibarted  yield")
plt.xlabel("Maturity")
plt.ylabel("Yield")
plt.legend()
plt.show()


yield_error = sum(abs(yield_curve - model_yield))
price_error = sum(abs(model_prices - bond_prices))

print2(price_error, yield_error)
