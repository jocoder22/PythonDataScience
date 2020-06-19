#!/usr/bin/env python
import numpy as np

nlist = [0.06,0.07, 0.08, 0.09, 0.10 ]
def forward_rate(spotList, tstar, t):
  nume = pow(1 + spotList[tstar -1], tstar)
  denom = pow(1 + spotList[t], t+1)
  result =  np.sqrt(denom / nume) - 1
  return round(result * 100, 2) 

print(forward_rate(nlist,1,2))

