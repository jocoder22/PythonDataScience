#!/usr/bin/env python

def print2(*args):
  for arg in args:
    print(arg, end="\n\n")

def spot_from_par(parList, year):
  """The spot_from_par function calculates the spot rate from the 
     par rate using bootstrapping method
     
     Inputs:
          parList (list) : list with par rates for increasing years
          year (int) : the year spot rate
          
     Output:
          rate (float) : the calculate n-year spot rate
  
  """
  price, n , total = 100, year, 0
  
  if n == 1:
    return parList[0]

  for ind, val in enumerate(parList, start=1):
    total += (parList[n-1]*price)/pow((1 + val),ind)
    
    if ind == n-1:
      rate = pow((parList[n-1]*price + price)/(price - total), 1/n) - 1
      return round(rate*100, 2)

    
par_rate = [0.06, 0.069673, 0.079050, 0.08811, 0.096855]
print2(spot_from_par(par_rate, 5))

splist = []
for n in range(1,6):
  sprate = spot_from_par(par_rate, n)
  splist.append(sprate)

print2(splist)
