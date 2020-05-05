import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize


# Set general parameters
s0 = 100
r = 0.1  # from govt bond prices and their yields

# set put parameters
# sigma and price are unknown
Kp = 100
Tp = 2

# set call parameters
Kc = 95
Tc = 2
price = 15


