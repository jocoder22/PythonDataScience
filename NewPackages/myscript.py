import os
import pandas as pd
import mypackage

path = "D:\PythonDataScience\importingData\localData"

os.chdir(path)
print(os.getcwd())

data = pd.read_csv("portfolios.csv", parse_dates = True, index_col= 0)


mypackage.print2(data)