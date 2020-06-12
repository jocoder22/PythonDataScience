#!/usr/bin/env python
import numpy as np
import pandas as pd
from decimal import *

Year = ["2008", "2009", "2010", "2011", "2012", "2013"]
FCFE_growth = [0.18, 0.18, 0.16, 0.12, 0.11, 0.06]
equity_discount_rate = 0.125


def ggm(dividend, dividend_growth_rate, equity_discount_r):
  """
  
  
  
  
  """
  
  pv_ggm = dividend*(1 + dividend_growth_rate)/(equity_discount_r -  dividend_growth_rate)
  
  return pv_ggm


def presentvalue(amt, rate, time):
  """
  
  
  
  
  """
  
  pv = amt/(1+ rate)**time
  
  return pv


def comp2(val, g, r, t):
  """





  """
  
  V0 = ggm(val, Decimal(g), Decimal(r))
  V = presentvalue(V0, Decimal(r), Decimal(t))
  
  return V


fcfe2007 = 2.0

fcfe = []
fcfe.append(fcfe2007 * (1+FCFE_growth[0]))

for i in range(len(FCFE_growth)-1):
  fcfe.append(round((fcfe[i] * (1+ FCFE_growth[i+1])),3))
  
pv_fcfe = [round(presentvalue(v, equity_discount_rate, 1+i),3) for i,v in enumerate(fcfe)]
  
print(fcfe)
print(pv_fcfe)

comp1 = sum(pv_fcfe)
print(comp1)

cc = comp2(fcfe[-1],g ,equity_discount_rate, 6)
value = comp1 + cc
print(value)



def multi_ggm(fcfe_list, dividend, g, r):
  """
  
  
  
  """
  
  n = len(fcfe_list)
  FCFE = []
  FCFE.append(dividend * (1+fcfe_list[0]))

  for i in range(n-1):
    FCFE.append(FCFE[i] * (1+ fcfe_list[i+1]))

  pv_fcfe = [presentvalue(v, equity_discount_rate, 1+i) for i,v in enumerate(FCFE)]
  
  
  comp1 = sum(pv_fcfe)
  comp2_ = comp2(FCFE[-1],g ,equity_discount_rate, n)
  value = comp1 + comp2_
  return round(value, 3)


def multi_ggm2(fcfe_list_, dividend_, g_, r_):
  """
  
  
  
  """
  from decimal import Decimal
  
  fcfe_list = list(map(Decimal, fcfe_list))
  dividend = Decimal(dividend_)
  g = Decimal(g_)
  r = Decimal(r_)
  n = Decimal(len(fcfe_list))
  
  fcfe_list = 
  FCFE = []
  FCFE.append(dividend * (1+fcfe_list[0]))

  for i in range(n-1):
    FCFE.append(FCFE[i] * (1+ fcfe_list[i+1]))

  # data = list(round(map(Decimal,FCFE),2))
  data= [round(Decimal(v), 2) for v in FCFE]

  pv_fcfe = [round(Decimal(presentvalue(v, r, 1+i)),2) for i,v in enumerate(FCFE)]
  
  
  comp1 = sum(pv_fcfe)
  
  # comp2_ = Decimal(comp2(FCFE[-1],g ,r, n))
  comp2_ = Decimal(comp2(data[-1],g ,r, n))
  value = comp1 + comp2_
  # value =  comp2_
  return round(value, 2), pv_fcfe,FCFE, data[-1]


print(multi_ggm(FCFE_growth,fcfe2007,g, equity_discount_rate))
