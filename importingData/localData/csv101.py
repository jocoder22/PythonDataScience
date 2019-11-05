import os
import pandas as pd 

sp = '\n\n'

def print2(*args):
    for obj in args:
        print(obj, end="\n\n")




params = {"sp":"\n\n", "end":"\n\n"}
# charge working directory
path = f"D:\PythonDataScience\importingData\localData"
os.chdir(path)

# load the csv file
data = pd.read_csv("portfolios.csv")
print(data.head())

colnames = list(data)
colnames2 = data.columns
print2(colnames, colnames2)

# import limited columns
limitcol = ['AMZN', 'AAPL', 'MSFT', 'AGG', 'JPM']
limitcol = [0, 4, 3, 6]
data = pd.read_csv("portfolios.csv", usecols=limitcol)
print2(data.head(), data.shape)

# import limited rows
data = pd.read_csv("portfolios.csv", usecols=limitcol, nrows = 200)
print2(data.head(), data.shape)


# skip some rows, must use header=None, and specific column names using names
data = pd.read_csv("portfolios.csv", skiprows=2000, 
                    nrows=300, header=None, names=colnames)

print2(data.head(), data.shape)