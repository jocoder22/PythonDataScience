#!/usr/bin/env python
import numpy as np
import pandas as pd

Year = ["2008", "2009", "2010", "2011", "2012", "2013"]
FCFE_growth = [0.18, 0.18, 0.16, 0.12, 0.11, 0.06]
equity_discount_rate = 0.125

def presentvalue(amt, rate, time):
  """
  
  
  
  
  """
  
  pv = amt/(1+ rate)^time
  
  return pv

fcfe2007 = 2.0

fcfe = []
fcfe.append(fcfe2007 * (1+FCFE_growth[0]))

for i in range(len(FCFE_growth)-1):
  fcfe.append(round((fcfe[i] * (1+ FCFE_growth[i+1])),3))
  
pv_fcfe = [round(presentvalue(v, equity_discount_rate, 1+i),3) for i,v in enumerate(fcfe)]
  
print(fcfe)
print(pv_fcfe)

comp1 = sum(pv_fcfe)

