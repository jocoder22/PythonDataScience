
def spot_from_par(parList, year):
  price, n = 100, year
  total = 0
  for ind, val in enumerate(parList, start=1):
    total += (parList[n-1]*price)/pow((1 + val),ind)
    if ind == n-1:
      rate = pow((parList[n-1]*price + price)/(price - total), 1/n) - 1
      # print(round(rate*100,2), price - total)
      return round(rate*100, 2)

print(spot_from_par(par_rate, 5))
