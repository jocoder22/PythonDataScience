import os
import pandas as pd 

# charge working directory
path = f"D:\PythonDataScience\importingData\localData"
os.chdir(path)

data = pd.read_csv("portfolios.csv")
print(data.head())