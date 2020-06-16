#!/usr/bin/env python
import numpy as np
import pandas as pd
from decimal import Decimal

def ggm(dividend, dividend_growth_rate, equity_discount_r):
  """The ggm function computes the single-peroid valuation using Gordon Growth Model (ggm)

    Inputs:
      dividend: the most recently paid dividend per share
      dividend_growth_rate: the periodic growth rate of dividends
      equity_discount_rate: the periodic equity discount rate

    Output:
      ppv_ggm: present value of share
  
  """
  
  pv_ggm = dividend*(1 + dividend_growth_rate)/(equity_discount_r -  dividend_growth_rate)
  
  return pv_ggm

def presentvalue(amt, rate, time):
  """The presentvalue function computes the present value of a future payment

    Inputs:
      amt: the amount to be paid in future
      rate: the interest rate
      time: the time period

    Output:
      pv: present value
    
    """
  
    pv = amt/(1 + rate)**time

    return pv


def comp2(val, g, r, t):
  """The comp2 function computes the present value of 
      the terminal dividend payment

    Inputs:
      val: the terminal value of dividend
      g: divident growth rate
      rate: the interest rate
      time: the time period

    Output:
      pv: present value of time dividend payment
    

    """
  
  V0 = ggm(val, Decimal(g), Decimal(r))
  V = presentvalue(V0, Decimal(r), Decimal(t))
  
  return V


def multi_ggm2(fcfe_list_, dividend_, g_, r_):
  """The ggm function computes the multi-peroid valuation using Gordon Growth Model (ggm)

    Inputs:
      fcfe_list_: the free cash flow to equity for each year
      dividend_ : the last dividend payment
      g_: the periodic growth rate of dividends
      equity_discount_rate: the periodic equity discount rate

    Output:
      mp_ggm: multi-period asset value
  

  """
 

    fcfe_list = list(map(Decimal, fcfe_list_)) 
    dividend = Decimal(dividend_)
    g = Decimal(g_)
    r = Decimal(r_)
    n = Decimal(len(fcfe_list))
    nn = len(fcfe_list)
    one = Decimal(1)

    FCFE = []
    FCFE.append(dividend * (one+fcfe_list[0]))

    for i in range(nn-1):
        FCFE.append(FCFE[i] * (one+ fcfe_list[i+1]))

    data= [round(Decimal(v), 2) for v in FCFE]

    pv_fcfe = [round(Decimal(presentvalue(v, r, 1+i)),2) for i,v in enumerate(FCFE)]

    comp1 = sum(pv_fcfe)

    comp2_ = Decimal(comp2(data[-1],g ,r, n))
    mp_ggm = comp1 + comp2_

    return round(mp_ggm, 2)


def Hmodel(dividend,h, gs, gl, equity_discount_rate, one=1):
  """The Hmodel function computes the multi-peroid valuation using the H Model (H-Model)

    Inputs:
      dividend_ : the last dividend payment
      h_: length of peroid for expected linear decline to sustainable long term growth rate
      gs: the short term growth rate
      gl: long term growth rate after h
      equity_discount_rate: the periodic equity discount rate

    Output:
      hvalue: multi-period asset value
  
  """

  # from decimal import Decimal
  
  save_locals = locals()
  
  for key, val in save_locals.items():
    key = Decimal(val)
    
  n1 = dividend*(one + gl)
  n2 = dividend*(h/2)*(gs - gl)
  m = equity_discount_rate - gl
  
  hvalue = round((n1 + n2)/m, 2)
  
  return hvalue


  
Year = ["2008", "2009", "2010", "2011", "2012", "2013"]
FCFE_growth = [0.18, 0.18, 0.16, 0.12, 0.11, 0.06]
equity_discount_rate = 0.125
fcfe2007 = 2.0
g = 0.06

print(multi_ggm2(FCFE_growth,fcfe2007,g, equity_discount_rate))


dividend = 2.50
n_years = 15
gs = 0.15
gl = 0.06
r = 0.10
print(Hmodel(dividend, n_years, gs, gl, r))











def ggm(dividend, dividend_growth_rate, equity_discount_r, one11):
  """The ggm function computes the single-peroid valuation using Gordon Growth Model (ggm)
    Inputs:
      dividend: the most recently paid dividend per share
      dividend_growth_rate: the periodic growth rate of dividends
      equity_discount_rate: the periodic equity discount rate
    Output:
      ppv_ggm: present value of share
  
  """
  
  pv_ggm = dividend*(one11 + dividend_growth_rate)/(equity_discount_r -  dividend_growth_rate)
  
  return pv_ggm

def presentvalue(amt, rate, time, one1):
  """The presentvalue function computes the present value of a future payment

    Inputs:
      amt: the amount to be paid in future
      rate: the interest rate
      time: the time period

    Output:
      pv: present value
    
    """
  
  pv = amt/(one1 + rate)**time

  return pv


def comp2(val, g, r, t, one22):
  """The comp2 function computes the present value of 
      the terminal dividend payment

    Inputs:
      val: the terminal value of dividend
      g: divident growth rate
      rate: the interest rate
      time: the time period

    Output:
      pv: present value of time dividend payment
    
  """
  
  V0 = ggm(val, g, r, one22)
  V = round(presentvalue(V0, r, t, one22), 2)
  
  return V


# def multi_ggm22(fcfe_list_, dividend_, g_, r_):
def multi_ggm2(fcfe_list, dividend, g, r):
  """The ggm function computes the multi-peroid valuation using Gordon Growth Model (ggm)

    Inputs:
      fcfe_list_: the free cash flow to equity for each year
      dividend_ : the last dividend payment
      g_: the periodic growth rate of dividends
      equity_discount_rate: the periodic equity discount rate

    Output:
      mp_ggm: multi-period asset value
  
  """
  # from decimal import Decimal
  
  save_locals = locals()
  
  listt = []
  
  for i, val in enumerate(save_locals.values()):
    if isinstance(val, list):
      listt.append(list(map(Decimal, val)))
    else:
      listt.append(Decimal(val))

  nn = len(fcfe_list)
  n = Decimal(nn)
  one = Decimal(1)

  # print(len(save_locals))
  # fcfe_list = list(map(Decimal, fcfe_list_)) 
  # dividend = Decimal(dividend_)
  # g = Decimal(g_)
  # r = Decimal(r_)
  # n = Decimal(len(fcfe_list))
  
  FCFE = []
  FCFE.append(listt[1] * (one + listt[0][0]))

  for i in range(nn-1):
    FCFE.append(FCFE[i] * (one + listt[0][i+1]))

  # data= [round(Decimal(v), 2) for v in FCFE]
  # print(FCFE)

  pv_fcfe = [round(presentvalue(v, listt[3], Decimal(i+1),one),2) for i,v in enumerate(FCFE)]

  # print(pv_fcfe)
  comp1 = sum(pv_fcfe)
  # print(comp1)

  comp2_ = comp2(round(FCFE[-1],2), listt[2],listt[3], n, one)
  # print(comp2_)
  mp_ggm = comp1 + comp2_

  return round(mp_ggm, 2)


  
Year = ["2008", "2009", "2010", "2011", "2012", "2013"]
FCFE_growth = [0.18, 0.18, 0.16, 0.12, 0.11, 0.06]
equity_discount_rate = 0.125
fcfe2007 = 2.0
g = 0.06

print(multi_ggm22(FCFE_growth,fcfe2007,g, equity_discount_rate))
