#!/usr/bin/env python
# Import required modules for this CRT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from printdescribe import print2, describe2, changepath

# import excel sheets
path = r"D:\Wqu_FinEngr\Portfolio Theory and Asset Pricing\GroupWork"

with changepath(path):
    data = pd.read_excel("GWP_PTAP_Data_2010.10.08.xlsx", skiprows=1, nrows = 13, 
                sheet_name='10 SPDRs and S&P 500', index_col=0)

describe2(data)
print2(data)


df_return = data.pct_change().dropna()
print2(df_return)


# df_activeReturn = df_return.sub(df_return.iloc[:,-1], axis=0).drop(['SP_500'], axis=1)
df_activeReturn = df_return.sub(df_return['S&P 500'], axis=0).drop(['S&P 500'], axis=1)
print2(df_activeReturn)



for col in df_return.columns[:-1]:
    plt.figure(figsize=[10, 8])
    plt.plot(df_return[[col, "S&P 500"]])
    plt.legend([col, "S&P500"])
    plt.show()

plt.figure(figsize=[10, 8])
plt.plot(df_return)
plt.legend(df_return.columns.tolist())
plt.show()