#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


np.random.seed(112)

s0= 40          # Today's stock price
K= 40           # Strike/Exercise price
T= 0.5          # Maturity (in years)
r= 0.05         # Risk-free rate
sigma= 0.2      # Annualized volatility
barrier = 42    # Barrier level