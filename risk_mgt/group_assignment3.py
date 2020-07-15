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

tracking_error = df_activeReturn.std()
mate_ = np.sqrt((df_activeReturn ** 2).sum()/df_activeReturn.shape[0])
print2(tracking_error, mate_)

# for col in df_return.columns[:-1]:
#     plt.figure(figsize=[10, 8])
#     plt.plot(df_return[[col, "S&P 500"]])
#     plt.legend([col, "S&P500"])
#     plt.title(f"Returns {col} vs S&P 500")
#     # plt.show()
#     plt.pause(2)
#     # plt.clf()
#     plt.close()

# plt.figure(figsize=[10, 8])
# plt.plot(df_return)
# plt.legend(df_return.columns.tolist())
# # plt.show()
# plt.pause(3)
# # plt.clf()
# plt.close()

dd = pd.concat([tracking_error, mate_], axis=1)
dd.columns = ["TrackingError", "Mean_Adj TrackingError"]

print2(dd)
