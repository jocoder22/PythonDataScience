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