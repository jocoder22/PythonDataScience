#!/usr/bin/env python
import numpy as np
import pandas as pd

Year = ["2008", "2009", "2010", "2011", "2012", "2013"]
FCFE_growth = [0.18, 0.18, 0.16, 0.12, 0.11, 0.06]

def presentvalue(amt, rate, time):
  """
  
  
  
  
  """
  
  pv = amt/(1+ rate)^time
  
  return pv

fcfe2007 = 2.0

fcfe = []
fcfe[0] = fcfe2007 * (1+FCFE_growth[0])
for i in range(1,len(FCFE_growth)):
  fcfe[i] = fcfe[i-1] * FCFE_growth[i-1]
  
print(fcfe)

