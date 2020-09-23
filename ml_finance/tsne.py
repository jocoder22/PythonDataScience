#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import datetime
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score

from printdescribe import print2, changepath

import warnings
warnings.filterwarnings('ignore')

# select required folder
patth = r"D:\PythonDataScience\ml_finance"

# import the necessary datasets
with changepath(patth):
    # dataset3.to_csv("assets3.csv",  compression='gzip')
    dataset3 = pd.read_csv("assets3.csv",  compression='gzip', parse_dates=True, index_col="Date")
    dataset2 = pd.read_csv("assets2.csv",  compression='gzip', parse_dates=True, index_col="Date")

# rename columns and drop s&p500
dataset2.rename(columns={"Adj Close": "SPX"}, inplace=True)
dataset3.drop(columns =["^GSPC"], inplace=True)


# combine the datasets
alldata = pd.concat([dataset3, dataset2], axis=1)
data2 = alldata.copy()
# data2 = data2.loc[:"2013-12-20", :]
print2(data2.iloc[:,:5].tail(), data2.shape, data2.iloc[:,-5:].tail())


def check__nulls(df):
    """
    Test and report number of NAs in each column of the input data frame
    :param df: pandas.DataFrame
    :return: None
    
    """
    for col in df.columns:
        _nans = np.sum(df[col].isnull())
        if _nans > 0:
            print(f'{_nans} NaNs in column {col}')
            
    print(f'New shape of df: {df.shape}')
    
    
check__nulls(data2)

# compute asset returns
asset_returns = np.log(data2 / data2.shift(1))
asset_returns = asset_returns.iloc[1:, :]
asset_returns.iloc[:, :n_stocks_show].head()

# Get the SPX time series. This now returns a Pandas Series object indexed by date
spx_index = asset_returns.loc[:, 'SPX']

short_rolling_spx = pd.core.series.Series(np.zeros(len(asset_prices.index)), index=asset_prices.index)
long_rolling_spx = short_rolling_spx

# Calculate the 20 and 100 days moving averages of log-returns
short_rolling_spx = spx_index.rolling(20).mean()
long_rolling_spx = spx_index.rolling(100).mean()


# Plot the index and rolling averages
fig=plt.figure(figsize=(12, 5), dpi= 80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1)
ax.plot(spx_index.index, spx_index, label='SPX Index')
ax.plot(short_rolling_spx.index, short_rolling_spx, label='20 days rolling')
ax.plot(long_rolling_spx.index, long_rolling_spx, label='100 days rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Log returns')
ax.legend(loc=2)
plt.show()


# Standardize features by removing the mean and scaling to unit variance
# Centering and scaling happen independently on each feature by computing the relevant statistics 
std_log_ret = StandardScaler().fit_transform(asset_returns.values)
std_log_ret = pd.DataFrame(data=std_log_ret, 
                              index=asset_returns.index,
                              columns=asset_returns.columns.values)

std_log_ret.iloc[:, :10].head()


# Calculate the 20 and 100 days moving averages of the log-returns
short_rolling_spx = std_log_ret[['SPX']].rolling(window=20).mean()
long_rolling_spx = std_log_ret[['SPX']].rolling(window=100).mean()


# Plot the index and rolling averages
fig=plt.figure(figsize=(12, 5), dpi= 80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1,1,1)
ax.plot(log_ret_df_std.index, log_ret_df_std[['SPX']], label='SPX Index')
ax.plot(short_rolling_spx.index, short_rolling_spx, label='20 days rolling')
ax.plot(long_rolling_spx.index, long_rolling_spx, label='100 days rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Log returns')
ax.legend(loc=2)
plt.show()
