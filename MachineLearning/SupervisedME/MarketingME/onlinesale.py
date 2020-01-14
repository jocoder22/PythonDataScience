import os
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

# import excel file from web
# Note that the output of pd.read_excel() is a Python dictionary with sheet names as keys 
# and corresponding DataFrames as corresponding values for multisheet dataset
# Returns a dataframe is sheet_name is specified

# data = pd.read_excel(url, sheet_name="Online Retail")
# data = pd.DataFrame(datat)
# print(data.keys(), data.columns, data.dtypes, data.head())

# print2(data.head())
# # saving as pickle file
mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
# data.to_pickle(os.path.join(mydir, "onlinedata.pkl"))


# load pickle file
onlinedata = pd.read_pickle(os.path.join(mydir, "onlinedata.pkl"))
print2(onlinedata.head(), onlinedata.shape)


