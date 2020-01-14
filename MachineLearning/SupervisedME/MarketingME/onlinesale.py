import os
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import datetime as dt
# from datetime import date as dt
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')
# plt.style.use('seaborn-whitegrid')

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
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
# data.to_pickle(os.path.join(mydir, "onlinedata.pkl"))


# load pickle file
onlinedata = pd.read_pickle(os.path.join(mydir, "onlinedata.pkl"))

# 'InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
#        'UnitPrice', 'CustomerID', 'Country'],
#       dtype='object')
onlinedata['InvoiceMonth'] = onlinedata['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month, 1))
customergroups = onlinedata.groupby('CustomerID')['InvoiceMonth']
onlinedata['CohortMonth'] = customergroups.transform('min')
onlinedata['CohortMonth2'] = onlinedata.groupby('CustomerID')['InvoiceDate'].transform('min')

# Extract year and month for each invoiceDate and cohort group
customerYear, customerMonth = onlinedata['InvoiceMonth'].dt.year, onlinedata['InvoiceMonth'].dt.month
cohortYear, cohortMonth =  onlinedata['CohortMonth'].dt.year, onlinedata['CohortMonth'].dt.month

yeardiff = customerYear - cohortYear
monthdiff = customerMonth - cohortMonth
onlinedata['cohortIndex'] = yeardiff * 12 + monthdiff + 1

# Anothe method to extract cohortIndex
onlinedata['cohortIndex2'] = onlinedata["InvoiceDate"] - onlinedata['CohortMonth2'] 
onlinedata['cohortIndex2'] = round(onlinedata['cohortIndex2']/np.timedelta64(1, "M") + 1)

cohortgroups = onlinedata.groupby(["CohortMonth", "cohortIndex"])
cohortNumber = cohortgroups["CustomerID"].apply(pd.Series.nunique).reset_index()


cohortTable = cohortNumber.pivot(index = "CohortMonth",
                                 columns = "cohortIndex",
                                 values = "CustomerID")

Retention = cohortTable.divide(cohortTable.iloc[:, 0], axis=0).round(3) 

print2(onlinedata.loc[:, "InvoiceDate": ].tail(), onlinedata.info(), onlinedata.columns, cohortNumber, cohortTable, Retention)
Retention.index = Retention.index.strftime("%Y-%m-%d")


plt.figure(figsize=(12, 10))
sns.heatmap(data = Retention,
            annot = True,
            fmt = '.0%',
            vmin = 0.0, vmax = 0.5,
            # cmap = "BuGn")
            )

# plt.yticks(rotation=0)
plt.title("Retention Ratio")
plt.show()

print2(Retention.info(), Retention.index)