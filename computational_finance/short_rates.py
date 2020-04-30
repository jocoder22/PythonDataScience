import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random

# set market parameters
r0 =  0.05
alpha = 0.2
b = 0.08
sigma = 0.025


#  define useful functions
def vasicef_mean(r, t1, t2):
  """
  
  """
  r0_discounted = r * np.exp(-alpha*(t2-t1))
  b_discounted = b * (1 - np.exp(-alpha*(t2-t1)))
  
  return r0_discounted + b_discounted

