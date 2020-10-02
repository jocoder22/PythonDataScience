#!/usr/bin/env python
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

# clean the datasets, remove NaN smartly
# Get a summary view of NaNs
oo = data2.isnull().sum()

# look at the distribution of NaN through Histogram
# bimodular distribution
plt.figure(figsize=[12, 7])
oo.hist()
plt.axhline(y=50, color='r', linestyle='-')
plt.gca().set(xlabel ="Number of NaNs", ylabel="Features", title=f"Dataset of with {data2.shape[1]} features")
plt.show();


# Retain columns with 96% of data are not NaN
data3 = data2.dropna(axis=1, thresh=int(data2.shape[0]*0.96))
print2(data3.shape, data3.isnull().sum().sum(),  data3.iloc[:,:10].head())

# Remove row with remaining NaNs
data4 = data3.dropna(axis=0)
print2(data4.shape, data4.isnull().sum().sum())
print2(f"alldata: {alldata.shape}", f"data6: {data4.shape}")

# View dataset
print2(f"Asset Adjusted Closing Prices shape: {data4.shape}", data4.iloc[:,-5:].head())

# copy the combned datasets
asset_prices = data4.copy()
n_stocks_show = 12
print('Asset prices shape', asset_prices.shape)
print2(asset_prices.iloc[:, :n_stocks_show].head())


# view the head and tail of the datasets
print('Last column contains SPX index prices:')
print2(asset_prices.iloc[:, -10:].head())
print('Last column contains SPX index prices:')
print(asset_prices.iloc[:, -10:].tail())

# Initiate dataset for return calculations
asset_returns = pd.DataFrame(data=np.zeros(shape=(len(asset_prices.index), asset_prices.shape[1])), 
                             columns=asset_prices.columns.values,
                             index=asset_prices.index)

# normed_returns = asset_returns
asset_returns = asset_prices.pct_change().dropna()

# normed_returns is pandas.DataFrame that should contain normalized returns
# normed_returns = asset_prices.pct_change().dropna()
_mean = asset_returns.mean()
_std = asset_returns.std()
normed_returns = (asset_returns - _mean)/_std

print2("#"*20)
print2(normed_returns.iloc[-5:, -10:].head())
print2(normed_returns.iloc[:, -10:].tail(5))


# train_end = datetime.datetime(2012, 3, 26) 
# get split date as 80% of the dataset for training
train_end = normed_returns.index.values[int(normed_returns.shape[0] * 0.8)]
print2(f"This the train end date: {train_end}")

# define training and testing datasets
df_train, df_test = None, None
df_raw_train, df_raw_test = None, None


# split dataset into train and test datasets
# splt normalized dataset
df_train = normed_returns[normed_returns.index <= train_end].copy()
df_test = normed_returns[normed_returns.index > train_end].copy()

# split dataset that is not normalized
df_raw_train = asset_returns[asset_returns.index <= train_end].copy()
df_raw_test = asset_returns[asset_returns.index > train_end].copy()

# view the shapes
print('Train dataset:', df_train.shape)
print('Test dataset:', df_test.shape)
print2(df_raw_test.head(), df_train.iloc[:, :10].head(), df_raw_test.iloc[:, :10].head())


# select the name of the stock, without the s&p500 index
stock_tickers = normed_returns.columns.values[:-1]
assert 'SPX' not in stock_tickers, "By accident included SPX index"

n_tickers = len(stock_tickers)
print2(n_tickers)


# initialize a PCA
pca = None
cov_matrix = pd.DataFrame(data=np.ones(shape=(n_tickers, n_tickers)), columns=stock_tickers)
cov_matrix_raw = cov_matrix

if df_train is not None and df_raw_train is not None:
    stock_tickers = asset_returns.columns.values[:-1]
    assert 'SPX' not in stock_tickers, "By accident included SPX index"

    ### START CODE HERE ### (≈ 2-3 lines of code)
    # calculate the covariance matrix and do pca on the covariance matrix
    cov_matrix = df_train[stock_tickers].cov()
    pca = PCA()
    pca.fit(cov_matrix) 
    
    # computing PCA on S&P 500 stocks, not normed covariance matrix
    cov_matrix_raw = df_raw_train[stock_tickers].cov()
    
    ### END CODE HERE ###
    
    cov_raw_df = pd.DataFrame({'Variance': np.diag(cov_matrix_raw)}, index=stock_tickers)

    # cumulative variance explained
    var_threshold = 0.8
    var_explained = np.cumsum(pca.explained_variance_ratio_)
    num_comp = np.where(np.logical_not(var_explained < var_threshold))[0][0] + 1  # +1 due to zero based-arrays
    print('%d components explain %.2f%% of variance' %(num_comp, 100* var_threshold))

print2(pca.explained_variance_ratio_.shape)


# Plot pca variance_explained
if pca is not None:
    bar_width = 0.9
    n_asset = int((1 / 10) * normed_returns.shape[1])
    x_indx = np.arange(n_asset)
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)
    
    # Eigenvalues are measured as percentage of explained variance.
    rects = ax.bar(x_indx, pca.explained_variance_ratio_[:n_asset], bar_width, color='deepskyblue')
    ax.set_xticks(x_indx + bar_width / 2)
    ax.set_xticklabels(list(range(n_asset)), rotation=45)
    ax.set_title('Percent variance explained')
    ax.legend((rects[0],), ('Percent variance explained by principal components',))
plt.show()


if pca is not None:
    projected = pca.fit_transform(cov_matrix)

# the first two eigen-portfolio weights# the fi 
# first component
# get the Principal components
pc_w = np.zeros(len(stock_tickers))
eigen_prtf1 = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_tickers)
if pca is not None:
    pcs = pca.components_

    # get the first components
    # normalized to 1
    pc_w = pcs[0] / pcs[0].sum(axis=0)
    # pc_w = pcs[249] / pcs[249].sum(axis=0)
    
    # form dataframe of normalise first component(squeezed to one dimension) and sort
    eigen_prtf1 = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_tickers)
    eigen_prtf1.sort_values(by=['weights'], ascending=False, inplace=True)
    print('Sum of weights of first eigen-portfolio: %.2f' % np.sum(eigen_prtf1))
    eigen_prtf1.plot(title='First eigen-portfolio weights', 
                     figsize=(12,6), 
                     xticks=range(0, len(stock_tickers),10), 
                     rot=45, 
                     linewidth=3)
plt.show()



pc_w = np.zeros(len(stock_tickers))
eigen_prtf2 = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_tickers)

if pca is not None:
    pcs = pca.components_
    
    # get the second component
    # normalized to 1 
    pc_w = pcs[1] / pcs[1].sum(axis=0)
    
    # form dataframe of normalise second component(squeezed to one dimension) and sort
    eigen_prtf2 = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_tickers)
    eigen_prtf2.sort_values(by=['weights'], ascending=False, inplace=True)
    print('Sum of weights of second eigen-portfolio: %.2f' % np.sum(eigen_prtf2))
    eigen_prtf2.plot(title='Second eigen-portfolio weights',
                     figsize=(12,6), 
                     xticks=range(0, len(stock_tickers),10), 
                     rot=45, 
                     linewidth=3)
plt.show()


def sharpe_ratio(ts_returns, periods_per_year=252):
    """
    sharpe_ratio - Calculates annualized return, annualized vol, and annualized sharpe ratio, 
                    where sharpe ratio is defined as annualized return divided by annualized volatility 
                    
    Arguments:
    ts_returns - pd.Series of returns of a single eigen portfolio
    
    Return:
    a tuple of three doubles: annualized return, volatility, and sharpe ratio
    """
    
    annualized_return = 0.
    annualized_vol = 0.
    annualized_sharpe = 0.
    
    # compute annaulized returns
    annualized_return = ts_returns.add(1).prod() ** (periods_per_year/ts_returns.shape[0]) - 1
    
    # compute annualized volatility
    annualized_vol = np.sqrt(periods_per_year*ts_returns.var())

    # compute annualized sharpe ratio
    annualized_sharpe = annualized_return / annualized_vol

    return annualized_return, annualized_vol, annualized_sharpe



if df_raw_test is not None:
    # get the first eigen portfolio return: returns dot portfolio weights (eigen values)
    eigen_prtf1_returns = np.dot(df_raw_test.loc[:, eigen_prtf1.index], eigen_prtf1 / 100)
    eigen_prtf1_returns = pd.Series(eigen_prtf1_returns.squeeze(), index=df_test.index)
    
    # compute annualized returns, volatility and sharpe ratio
    er, vol, sharpe = sharpe_ratio(eigen_prtf1_returns)
    print('First eigen-portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (er*100, vol*100, sharpe))
    year_frac = (eigen_prtf1_returns.index[-1] - eigen_prtf1_returns.index[0]).days / 252

    df_plot = pd.DataFrame({'PC1': eigen_prtf1_returns, 'SPX': df_raw_test.loc[:, 'SPX']}, index=df_test.index)
    np.cumprod(df_plot + 1).plot(title='Returns of the market-cap weighted index vs. First eigen-portfolio', 
                             figsize=(12,6), linewidth=3)
plt.show()


if df_raw_test is not None:
    # get the second eigen portfolio return: returns dot portfolio weights (eigen values)
    eigen_prtf2_returns = np.dot(df_raw_test.loc[:, eigen_prtf2.index], eigen_prtf2 / 100)
    eigen_prtf2_returns = pd.Series(eigen_prtf2_returns.squeeze(), index=df_test.index)
    
    # compute annualized returns, volatility and sharpe ratio
    er, vol, sharpe = sharpe_ratio(eigen_prtf2_returns)
    print('Second eigen-portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (er*100, vol*100, sharpe))


# n_portfolios = 120
n_portfolios = n_tickers
annualized_ret = np.array([0.] * n_portfolios)
sharpe_metric = np.array([0.] * n_portfolios)
annualized_vol = np.array([0.] * n_portfolios)
idx_highest_sharpe = 0 # index into sharpe_metric which identifies a portfolio with rhe highest Sharpe ratio
    
if pca is not None:
    for ix in range(n_portfolios):
        
        # get normalised component, portfolio weights
        pc_w = pcs[:,ix] / np.sum(pcs[:, ix])
        
        # form dataframe and sort
        eigen = pd.DataFrame(data ={'weights': pc_w.squeeze()}, index = stock_tickers) 
        
        # get the eigen portfolio return: returns dot portfolio weights (eigen values)
        eigen_returns = pd.Series(np.dot(df_raw_test.loc[:, eigen.index], eigen).squeeze(), index=df_test.index)
        
        # compute annualized returns, volatility and sharpe ratio
        annualized_ret[ix], annualized_vol[ix], sharpe_metric[ix] = sharpe_ratio(eigen_returns)
    
    # find portfolio with the highest Sharpe ratio 
    # change lareg sharpe ratio to NaN
    sharpe_metric[sharpe_metric >= 100.0] = np.nan
    
    # Zero all values of NaN, positive and negative infinity
    np.nan_to_num(sharpe_metric, nan=0, posinf=0, neginf=0, copy = False)
    
    # find the index of the highest values that's not NaN
    idx_highest_sharpe = np.nanargmax(sharpe_metric)
    print2(f"Max sharpe ration index: {idx_highest_sharpe}")                                       
    
    # print out eigen porfolio with highest sharpe ratio
    print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' % 
          (idx_highest_sharpe,
           annualized_ret[idx_highest_sharpe]*100, 
           annualized_vol[idx_highest_sharpe]*100, 
           sharpe_metric[idx_highest_sharpe]))
    
    # plot the sharpe ratio
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)
    ax.plot(sharpe_metric, linewidth=3)
    ax.set_title('Sharpe ratio of eigen-portfolios')
    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('Portfolios')

plt.show()


results = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol, 'Sharpe': sharpe_metric})
results.dropna(how = 'any', inplace = True)
results.sort_values(by=['Sharpe'], ascending=False, inplace=True)
print2(results.head(10))
