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

df = onlinedata.groupby('CustomerID').agg(
                    Recency = ('InvoiceDate', lambda x : (refDate - x.max()).days),
                    Frequency = ('InvoiceNo', 'count'),
                    MonetaryValue = ('Totalcost', 'sum'))


_ = onlinedata[onlinedata["CustomerID"] == 12346.0]
print2(onlinedata, df, _ )

nlabels = list(range(4, 0, -1))
dlabels = list(range(1,5))


df['R'] = pd.qcut(df.Recency, q=4, labels=nlabels)
df['F'] = pd.qcut(df.Frequency, q=4, labels=dlabels)
df['M'] = pd.qcut(df.MonetaryValue, q=4, labels=dlabels)

df['rfm_Segment'] = df.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']), axis=1)
df['rfm_Score'] = df.loc[:, 'R':'M'].sum(axis=1)

def custLevels(dataa, col):
    if dataa[col] < 6 :  return "LowClass"
    elif dataa[col] < 10 : return "MiddleClass"
    else: return "TopClass"
    
df['custCat'] = df.apply(custLevels, args=['rfm_Score'], axis=1)

# cust_Stats = df.groupby('custCat').agg(
#                 Recency_mean = ('Recency', 'mean'),
#                 Frequency_mean = ('Frequency', 'mean'),
#                 MonetaryValue_mean = ('MonetaryValue', 'mean'),
#                 MonetaryValue_count  = ('MonetaryValue', 'count')
# )


# 2nd method for multilevel columns
cust_Stats = df.groupby('custCat').agg({
                'Recency' : 'mean',
                'Frequency': 'mean',
                'MonetaryValue': ['mean', 'count'] })

cust_Stats.columns = ["_".join(yy) for yy in cust_Stats.columns.ravel()]

print2(onlinedata, df, _ , cust_Stats)

# saving as pickle file
mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
df.to_pickle(os.path.join(mydir, "RFM.pkl"))
