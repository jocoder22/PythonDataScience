import os
import pandas as pd 

sp = '\n\n'

params = {sp:"\n\n", end:"\n\n"}
# charge working directory
path = f"D:\PythonDataScience\importingData\localData"
os.chdir(path)

# load the csv file
data = pd.read_csv("portfolios.csv")
print(data.head())

colnames = list(data)
colnames2 = data.columns
print(colnames, colnames2, sep=sp)

# import limited columns
limitcol = ['AMZN', 'AAPL', 'MSFT', 'AGG', 'JPM']
limitcol = [0, 4, 3, 6]
data = pd.read_csv("portfolios.csv", usecols=limitcol)
# colnames = names(data)
print(data.head(), data.shape, sep=sp)

# import limited rows
data = pd.read_csv("portfolios.csv", usecols=limitcol, nrows = 200)
# colnames = names(data)
print(data.head(), data.shape, sep=sp)