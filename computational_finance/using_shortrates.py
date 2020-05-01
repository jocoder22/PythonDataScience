#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from short_rates import vasicek_mean, vasicek_var

# Analytical bond price (ZCB)
def a_part(t1,t2):
  """
  
  """
  numer_ = 1 - np.exp(-alpha * (t2-t1))
  
  result = numer_/alpha
  
  return result


def d_part(t1, t2):
  """
  
  
  
  """
  
  val1 = b - (sigma**2/(2*alpha**2))
  val2 = a_part(t1,t2) - (t2 - t1)
  val3 = sigma**2 * a_part(t1,t2)**2 / (4*alpha)
  
  result = val1 * val2 - val3
  
  return result
  
def bond_price(r,t,T):
  """
  
  
  """
  bondprice = np.exp(-a_part(t,T)*r*d_part(t,T))
  
  return bondprice
