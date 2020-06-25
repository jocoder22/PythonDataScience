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


indexes = pd.DataFrame(index="A B C D E".split())
indexes["PricePerShare_0"], indexes["Return"] = [60, 30, 15, 12, 4.80], [0.095, -0.1140, -0.342, 0.38, 0.475]
indexes["PricePerShare_1"],indexes["DividendPerShare"] = [65.7, 26.58, 9.87, 16.56, 7.08], [1.05, 0.14, 0.00, 0.07, 0.00]
indexes["SharesOutstanding"] = [3300, 11000, 5500, 8800, 7700]
indexes["FloatAdjFactor"] = [1.0, 0.7, 0.9, 0.25, 0.8]
indexes["MarketFloat"] = indexes["FloatAdjFactor"].mul(indexes["SharesOutstanding"])
