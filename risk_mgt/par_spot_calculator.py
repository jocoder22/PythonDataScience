
def print2(*args):
  for arg in args:
    print(arg, end="\n\n")

def spot_from_par(parList, year):
  price, n = 100, year
  total = 0
  for ind, val in enumerate(parList, start=1):
    total += (parList[n-1]*price)/pow((1 + val),ind)
    if ind == n-1:
      rate = pow((parList[n-1]*price + price)/(price - total), 1/n) - 1
      # print(round(rate*100,2), price - total)
      return round(rate*100, 2)

    
par_rate = [0.06, 0.069673, 0.079050, 0.08811, 0.096855]
print(spot_from_par(par_rate, 5))
