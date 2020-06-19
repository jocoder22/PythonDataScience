#!/usr/bin/env python
import numpy as np

nlist = [0.06,0.07, 0.08, 0.09, 0.10 ]
def forward_rate(spotList, tstar, t):
  """The forward_rate function calculates the forward rate e.g f(T*, T)
      given yearly spot rates.
      
      Inputs:
          spotList (list) : yearly spot rates
          tstar (int): time until initiation of the rate
          t (int): the tiemto maturity from tstar
          
       Output:
            result (float): forward rate
  
  """
  
  nume = pow(1 + spotList[tstar -1], tstar)
  denom = pow(1 + spotList[t+tstar - 1], t+tstar)
  result =  pow((denom / nume), 1/tstar) - 1
  return round(result * 100, 2) 

print(forward_rate(nlist,1,2))

