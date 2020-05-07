#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as opt


# create yield curve for integer maturities
years = np.linspace(1,10,10)

# We suppose the yield curve for maturities from 1 to 10 yrs takes the form
# y(t) = 1/75 * (t)**1/5 + 0.04
yield_curve = (years)**(1/5)/75 + 0.04
bond_prices = np.exp(-yield_curve*year)


# plot yield curve
plt.plot(years, yield_curve*100)
plt.xlabel("Maturity")
plt.show
