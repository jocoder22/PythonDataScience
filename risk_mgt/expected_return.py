#!/usr/bin/env python
import numpy as np
import pandas as pd
from printdescribe import print2, describe2, changepath

S0 = [[120, 90], [140, 90], [130,100]]
prob = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

def return_var(prices, probb):
  r = [np.dot(np.array(i[0]), np.array(i[1])) for i in zip(prices, probb)]
  t = [np.sqrt(np.dot(np.array(i[2]), np.subtract(np.array(i[0]), np.array(i[1]))**2)) for i in zip(prices, r, probb)]
  print2(r, t)
  return r, t

vars = return_var(S0, prob)
print2(S0, prob, vars)
