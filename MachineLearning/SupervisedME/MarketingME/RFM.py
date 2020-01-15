import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')
# plt.style.use('seaborn-whitegrid')

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"

# load pickle file
onlinedata = pd.read_pickle(os.path.join(mydir, "onlinedata.pkl"))

# 'InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
#        'UnitPrice', 'CustomerID', 'Country'],
#       dtype='object')
refDate = max(onlinedata.InvoiceDate) + dt.timedelta(days=1)
onlinedata["Totalcost"] = onlinedata["Quantity"].mul(onlinedata["UnitPrice"], axis=0)


