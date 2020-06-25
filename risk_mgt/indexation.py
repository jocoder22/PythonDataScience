import numpy as np
import pandas as pd
import tabulate

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
indexes.index.name = "Security"

indexes["PricePerShare_0"], indexes["Return"] = [60, 30, 15, 12, 4.80], [0.095, -0.1140, -0.342, 0.38, 0.475], 
indexes["PricePerShare_1"] = indexes["PricePerShare_0"] *  (1 + indexes["Return"])
indexes["DividendPerShare"] = [1.05, 0.14, 0.00, 0.07, 0.00]
indexes["SharesOutstanding"] = [3300, 11000, 5500, 8800, 7700]
indexes["FloatAdjFactor"] = [1.0, 0.7, 0.9, 0.25, 0.8]
indexes["MarketFloat"] = indexes["FloatAdjFactor"].mul(indexes["SharesOutstanding"])


indexes.reset_index(inplace=True)
print(tabulate.tabulate(indexes, headers=indexes.columns, tablefmt="fancy_grid", showindex="never"))



## Price Weighted index
## Time 0
pwi = pd.DataFrame(index=indexes.Security)

pwi["SharesIn_index"] = 1.0
pwi["PricePerShare_0"] = indexes["PricePerShare_0"].values
pwi["Value_0"] = pwi["SharesIn_index"].mul(pwi["PricePerShare_0"])
pwi["Weight_0"] = pwi["PricePerShare_0"] / pwi["Value_0"].sum() 

divisor = pwi.shape[0]
indexValue = pwi["Value_0"].sum() / divisor
print2(indexValue)

pwi.reset_index(inplace=True)
print(tabulate.tabulate(pwi, headers=pwi.columns, tablefmt="fancy_grid", showindex="never"))


## Price Weighted index
## Time 1
pwi_ = pd.DataFrame(index=indexes.Security)
pwi_["SharesIn_index"] = 1.0
pwi_["PricePerShare_1"] = indexes["PricePerShare_1"].values
pwi_["Value_1"] = pwi_["SharesIn_index"].mul(pwi_["PricePerShare_1"])
pwi_["Weight_1"] = pwi_["PricePerShare_1"] / pwi_["Value_1"].sum()


divisor = pwi_.shape[0]
indexValue = pwi_["Value_1"].sum() / divisor
print2(indexValue)


## Price Weighted index
## Total index return = income returns + price return
pwi_t = pd.DataFrame(index=indexes.Security)
pwi_t["SharesIn_index"] = 1.0
pwi_t["PricePerShare_1"] = indexes["PricePerShare_1"].values
pwi_t["DividendPerShare"] = indexes["DividendPerShare"].values
pwi_t["Value_1"] = pwi_t["SharesIn_index"].mul(pwi_t.loc[:,["PricePerShare_1", "DividendPerShare"]].sum(axis=1))
pwi_t["Weight_1"] = pwi_t["PricePerShare_1"] / pwi_t["Value_1"].sum()
print2(pwi_t)

divisor = pwi_t.shape[0]
indexValue3 = pwi_t["Value_1"].sum() / divisor

index_return2 = indexValue3/indexValue - 1
print2(indexValue3, round(index_return2*100, 2))


pwi_t.reset_index(inplace=True)
print(tabulate.tabulate(pwi_t, headers=pwi_t.columns, tablefmt="fancy_grid", showindex="never"))


