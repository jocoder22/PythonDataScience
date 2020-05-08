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
