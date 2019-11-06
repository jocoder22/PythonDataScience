import os
import pandas as pd 

sp = '\n\n'

def print2(*args):
    for obj in args:
        print(obj, end="\n\n")



def describe2(x):
    print2(x.head(), x.shape, x.info())


params = {"sp":"\n\n", "end":"\n\n"}
# charge working directory
path = f"D:\PythonDataScience\importingData\localData"
os.chdir(path)

# load the excel file
data = pd.read_excel("portfolioxx.xlsx")
print(data.head())

# select rows and columns
names = ["AMZN", "MSFT", "AGG", "VNQ"]
data = pd.read_excel("portfolioxx.xlsx", nrows=1000, skiprows=100, usecols="A,C,E:F", names=names)
print(data.head())
print2(data)


# import multiple sheets
data = pd.read_excel("portfolioSheet.xlsx",  sheet_name=None)
for sheetname, obj in data.items():
    print(sheetname, type(obj), end=sp)


# append multiple columns
stocks = pd.DataFrame()
name = ["Open", "High", "Low", "Close", "Volume", "Adjusted"]
for shname, ddict in data.items():
    ddict.columns = name
    ddict.insert(0, "Stockname", shname)
    ddict.insert(1, "OC_diff", ddict["Open"] - ddict["Close"])
    stocks = stocks.append(ddict)

# new_order = [-1,0,1,2,3,4,5]
# stocks = stocks[stocks.columns[new_order]]
# print2(stocks)

# stocks.insert(0, "NewAdj", stocks.Adjusted)
print2(stocks)