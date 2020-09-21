import os
import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from printdescribe import print2, changepath

import warnings
warnings.filterwarnings('ignore')

patth = r"D:\PythonDataScience\ml_finance"

with changepath(patth):
    # dataset3.to_csv("assets3.csv",  compression='gzip')
    dataset3 = pd.read_csv("assets3.csv",  compression='gzip', parse_dates=True, index_col="Date")
    dataset2 = pd.read_csv("assets2.csv",  compression='gzip', parse_dates=True, index_col="Date")

dataset2.rename(columns={"Adj Close": "SPX"}, inplace=True)
dataset3.drop(columns =["^GSPC"], inplace=True)

alldata = pd.concat([dataset3, dataset2], axis=1)
data2 = alldata.copy()
print2(data2.iloc[:,:5].tail(), data2.shape, data2.iloc[:,-5:].tail())

# clean the datasets, remove NaN smartly
# Get a summary view of NaNs
oo = data2.isnull().sum()

# look at the distribution through Histogram
# bimodular distribution
plt.figure(figsize=[12, 7])
oo.hist()
plt.axhline(y=50, color='r', linestyle='-')
plt.gca().set(xlabel ="Number of NaNs", ylabel="Features", title=f"Dataset of with {data2.shape[1]} features")
plt.show();

# Retain columns with 96% of data
data3 = data2.dropna(axis=1, thresh=int(data2.shape[0]*0.96))
print2(data3.shape, data3.isnull().sum().sum(),  data3.iloc[:,:10].head())

# Remove row with remaining NaNs
data4 = data3.dropna(axis=0)
print2(data4.shape, data4.isnull().sum().sum())
print2(f"alldata: {alldata.shape}", f"data6: {data4.shape}")

# View dataset
print2(f"Asset Adjusted Closing Prices shape: {data4.shape}", data4.iloc[:,-5:].head())

asset_prices = data4.copy()
n_stocks_show = 12
print('Asset prices shape', asset_prices.shape)
print2(asset_prices.iloc[:, :n_stocks_show].head())


print('Last column contains SPX index prices:')
print2(asset_prices.iloc[:, -10:].head())

print('Last column contains SPX index prices:')
print(asset_prices.iloc[:, -10:].tail())

asset_returns = pd.DataFrame(data=np.zeros(shape=(len(asset_prices.index), asset_prices.shape[1])), 
                             columns=asset_prices.columns.values,
                             index=asset_prices.index)

normed_returns = asset_returns
asset_returns = asset_prices.pct_change().dropna()

# normed_returns is pandas.DataFrame that should contain normalized returns
normed_returns = asset_prices.pct_change().dropna()
_mean = normed_returns.mean()
_std = normed_returns.std()
normed_returns = (normed_returns - _mean)/_std

print2(normed_returns.iloc[-5:, -10:].head())
print2(normed_returns.iloc[:, -10:].tail(5))


# train_end = datetime.datetime(2012, 3, 26) 
train_end = normed_returns.index.values[int(normed_returns.shape[0] * 0.7)]
print2(f"This the train end date: {train_end}")

df_train = None
df_test = None
df_raw_train = None
df_raw_test = None



df_train = normed_returns[normed_returns.index <= train_end].copy()
df_test = normed_returns[normed_returns.index > train_end].copy()

df_raw_train = asset_returns[asset_returns.index <= train_end].copy()
df_raw_test = asset_returns[asset_returns.index > train_end].copy()

print('Train dataset:', df_train.shape)
print('Test dataset:', df_test.shape)

print2(df_raw_test.head(), df_train.iloc[:, :10].head(), df_raw_test.iloc[:, :10].head())