import numpy as np
import pandas as pd

PriceReturn = [5.2, 4.1, -2.6]
IncomeReturn = [1.5, 1.5, 1.5]


def print2(*args):
  for arg in args:
    print(arg, end="\n\n")

PriceReturn = [5.2, 4.1, -2.6]
IncomeReturn = [1.5, 1.5, 1.5]

data = pd.DataFrame({"PriceIncome":PriceReturn, "IncomeReturn":IncomeReturn})*0.01

data["TotalReturn"] = data.sum(axis = 1)

v0 = 100
ivalue = v0 * (1 + data["PriceIncome"]).cumprod()
itotal = (v0 * (1 + data["TotalReturn"]).cumprod()).iloc[-1]

print2(data, round(ivalue.iloc[-1], 4), round(itotal, 4))

