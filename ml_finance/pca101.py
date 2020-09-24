
import os
import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
import pandas_datareader.data as pdr
from printdescribe import print2, changepath

import warnings
warnings.filterwarnings('ignore')

symbols = ['A', 'AA', 'AAPL', 'ABC', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 
           'AGN', 'AIG', 'AIV', 'AKAM', 'AKS', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMT', 'AMZN', 'AN', 'ANDV', 'ANF', 
           'AON', 'APA', 'APC', 'APD', 'APH', 'APOL', 'ARG', 'ATGE', 'AVB', 'AVY', 'AXP', 'AZO', 'BA', 'BAC', 
           'BAX', 'BBT', 'BBY', 'BCR', 'BDX', 'BEN', 'BHGE', 'BIIB', 'BK', 'BLL', 'BMY', 'BRCM', 'BSX', 'BXP', 
           'C', 'CA', 'CAG', 'CAH', 'CAM', 'CAT', 'CB', 'CBS', 'CCL', 'CELG', 'CERN', 'CHK', 'CHRW', 'CI', 'CINF',
           'CL', 'CLF', 'CLX', 'CMA', 'CMCSA', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 'COG', 'COP', 'COST', 'CPB', 
           'CSCO', 'CSX', 'CTAS', 'CTL', 'CTSH', 'CTXS', 'CVS', 'CVX', 'D', 'DE', 'DF', 'DGX', 'DHI', 'DHR', 'DIS', 
           'DNR', 'DO', 'DOV', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 
           'EMN', 'EMR', 'EOG', 'EQR', 'EQT', 'ES', 'ESRX', 'ETFC', 'ETN', 'ETR', 'EXC', 'EXPD', 'F', 'FAST', 'FCX',
           'FDO', 'FDX', 'FE', 'FFIV', 'FHN', 'FII', 'FISV', 'FITB', 'FLIR', 'FLS', 'FMC', 'FOXA', 'FRX', 'FTR', 'GD',
           'GE', 'GHC', 'GILD', 'GIS', 'GLW', 'GPC', 'GPS', 'GS', 'GT', 'GWW', 'HAL', 'HAR', 'HAS', 'HBAN', 'HCBK', 
           'HCN', 'HCP', 'HD', 'HES', 'HIG', 'HOG', 'HON', 'HOT', 'HP', 'HPQ', 'HRB', 'HRL', 'HRS', 'HST', 'HSY', 
           'HUM', 'IBM', 'IFF', 'INTC', 'INTU', 'IP', 'IPG', 'IR', 'IRM', 'ITW', 'IVZ', 'JBL', 'JCI', 'JEC', 'JNJ',
           'JNPR', 'JPM', 'JWN', 'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR', 'KSS', 'L', 'LB', 'LEG', 'LEN', 
           'LH', 'LLL', 'LLTC', 'LLY', 'LM', 'LMT', 'LNC', 'LOW', 'LUK', 'LUV', 'M', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP',
           'MCK', 'MCO', 'MDT', 'MKC', 'MMC', 'MMM', 'MO', 'MRK', 'MRO', 'MS', 'MSFT', 'MSI', 'MTB', 'MU', 'MUR', 'MWW',
           'MYL', 'NBL', 'NBR', 'NE', 'NEE', 'NEM', 'NFX', 'NI', 'NKE', 'NOC', 'NOV', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA',
           'NWL', 'OI', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PCAR', 'PCG', 'PCL', 'PCLN', 'PCP', 'PDCO',
           'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PHM', 'PKI', 'PNC', 'PNW', 'PPG', 'PPL', 'PSA', 'PWR', 'PX', 'PXD',
           'QCOM', 'QLGC', 'RAI', 'RF', 'RHI', 'RHT', 'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RSG', 'RTN', 'SBUX', 'SCG', 'SCHW',
           'SEE', 'SHW', 'SJM', 'SLB', 'SNA', 'SNDK', 'SO', 'SPG', 'SPGI', 'SRCL', 'SRE', 'STI', 'STR', 'STT', 'STZ', 'SUNEQ',
           'SWK', 'SWN', 'SYK', 'SYMC', 'SYY', 'T', 'TAP', 'TGT', 'TIF', 'TJX', 'TMK', 'TMO', 'TROW', 'TRV', 'TSN', 'TSS',
           'TWX', 'TXN', 'TXT', 'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'USB', 'UTX', 'VAR', 'VFC', 'VIAV', 'VLO', 'VMC', 'VNO',
           'VRSN', 'VTR', 'VZ', 'WAT', 'WBA', 'WDC', 'WEC', 'WFC', 'WFM', 'WHR', 'WM', 'WMT', 'WY', 'XEL', 'XL', 'XLNX', 'XOM',
           'XRAY', 'XRX', 'YUM', 'ZION', '1255459D', '1284849D', '1431816D', '1436513D', '1448062D', '1500785D', '1519128D',
           '1541931D', '9876544D', '9876566D', '9876641D', 'ATI', 'AVP', 'BBBY', 'BF/B', 'BIG', 'BMS', 'BRK/B', 'CSC', 'CVC',
           'DD', 'DOW', 'EMC', 'HSH', 'ITT', 'JCP', 'LXK', 'MDP', 'NYT', 'ODP', 'PBI', 'PLL', 'R', 'RDC', 'RRD', 'RSHCQ', 'SIAL',
           'SLM', 'SPLS', 'STJ', 'SVU', 'SWY', 'TEG', 'TER', 'TGNA', 'THC', 'X', 'MAR.1', '^GSPC']


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)
# pd.options.display.max_seq_items = None

patth = r"D:\PythonDataScience\ml_finance"
import pandas_datareader as pdr


start_date = '2000-01-01'
# end_date = '2019-12-31'
# pdr.get_data_yahoo(symbol, starttime, endtime)[['Adj Close']]
# dataset2 = pdr.get_data_yahoo("^GSPC", start_date)[['Adj Close']]
# dataset3 = pdr.get_data_yahoo(symbols, start=start_date)['Adj Close']

# print2(dataset2.head())

with changepath(patth):
    # dataset3.to_csv("assets3.csv",  compression='gzip')
    dataset3 = pd.read_csv("assets3.csv",  compression='gzip', parse_dates=True, index_col="Date")
    dataset2 = pd.read_csv("assets2.csv",  compression='gzip', parse_dates=True, index_col="Date")
    # datasets = pd.read_csv("assets.csv",  compression='gzip', index_col=0)
    # datasets, dataset2 = pd.read_csv(["assets.csv","assets2.csv"])
 
dataset2.rename(columns={"Adj Close": "SPX"}, inplace=True)
dataset3.drop(columns =["^GSPC"], inplace=True)

# # df.set_index('Date', inplace=True)
# print2(dataset2.head())
# datasets.reindex(dataset2.index)

alldata = pd.concat([dataset3, dataset2], axis=1)
data2 = alldata.copy()
data2 = data2.loc[:"2013-12-20", :]
print2(data2.iloc[:,:5].tail(), data2.shape)
# tt = "https://dumbstockapi.com/stock?format=tickers-only&exchange=NYSE"
# pp = pd.read_json(tt)
# pp = list(pp.values.ravel())

# download data and view
# data2 = dr.DataReader(pp, data_source='yahoo', start=start_date)['Adj Close']
# print2(f"Asset Adjusted Closing Pices shape: {data2.shape}", data2.iloc[:,:10].head())

# drop columns with NaN
data2.dropna(axis=1)

print(data2.iloc[:, :5].head())
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

# Retain columns with 99.5% of data
data5 = data2.dropna(axis=1, thresh=int(data2.shape[0]*0.96))
print2(data5.shape, data5.isnull().sum().sum(),  data5.iloc[:,:10].head())

# Remove row with remaining NaNs
data6 = data5.dropna(axis=0)
print2(data6.shape, data6.isnull().sum().sum())
print2(f"alldata: {alldata.shape}", f"data6: {data6.shape}")

# View dataset
print2(f"Asset Adjusted Closing Prices shape: {data6.shape}", data6.iloc[:,:10].head())


# _returns = pd.DataFrame(data=np.zeros(shape=(len(data6.index), data6.shape[1])), 
#                              columns=data6.columns.values,
#                              index=data6.index)

# norm_returns = _returns
# _returns = data6.pct_change().dropna()

# compute normalized returns
_returns = data6.pct_change().dropna()
_returns_mean = _returns.mean()
_returns_std = _returns.std()
norm_returns = (_returns - _returns_mean) / _returns_std

print(norm_returns.iloc[:, :5].head())

# veiw normalised returns
print2(norm_returns.iloc[:, -10:].head(), norm_returns.shape)



# split data into train and test datasets, based on timestamps
# 70% for the train dataset and 30% for the test dataset
indy_ = norm_returns.index.values[int(norm_returns.shape[0] * 0.7)]

# set memory space for efficient and fast programming
X_norm_train, X_norm_test = None, None
X_train, X_test = None, None

# split data into train and test datasets
X_norm_train = norm_returns[norm_returns.index <= indy_].copy()
X_norm_test = norm_returns[norm_returns.index > indy_].copy()
X_train_raw = _returns[_returns.index <= indy_].copy()
X_test_raw = _returns[_returns.index > indy_].copy()

# view the datasets
print2(X_norm_train.shape, X_norm_test.shape, X_train_raw.shape, X_test_raw.shape)
print2(X_norm_train.iloc[:,:5].head(), X_norm_test.iloc[:,:5].head(), 
       X_train_raw.iloc[:,:5].head(), X_test_raw.iloc[:,:5].head())

# get the stock tickers
stock_symbols = X_norm_train.columns.values[:-1]
num_ticker = len(stock_symbols)
print(num_ticker)

# intialise a pca and empty dataframe for the matrix
pca = PCA()
assert 'SPX' not in stock_symbols, "By accident included SPX index"

cov_mat = pd.DataFrame(data=np.ones(shape=(num_ticker, num_ticker)), columns=stock_symbols)
cov_matraw = cov_mat

cov_mat = X_norm_train[stock_symbols].cov()
cov_matraw = X_train_raw[stock_symbols].cov()

pca.fit(cov_mat)

cov_raw_df = pd.DataFrame({"variance":np.diag(cov_matraw)}, index=stock_symbols)
print2(cov_raw_df.iloc[:,:10].head())


var_threshold = [0.7, 0.8, 0.9, 0.95]
ncomp = len(pca.explained_variance_ratio_)
var_explained = np.cumsum(pca.explained_variance_ratio_)
for i in var_threshold:
    num_comp = np.where(np.logical_not(var_explained < i))[0][0] + 1
    print(f'{num_comp} components (about {np.round(num_comp/ncomp *100, 2)}%) explain {100* i}% of variance')
           
print2(len(pca.explained_variance_ratio_), pca.explained_variance_ratio_.shape)   


# bar_width = 0.9
n_asset = int((1/20) * norm_returns.shape[1])
x_indx = np.arange(n_asset)
fig, ax = plt.subplots()
fig.set_size_inches(14, 5)

# Eigenvalues are measured as percentage of explained variance.
# rects = ax.bar(x_indx, pca.explained_variance_ratio_[:n_asset], bar_width, color='deepskyblue')
rects = ax.bar(x_indx, pca.explained_variance_ratio_[:n_asset],  color='deepskyblue')
# ax.set_xticks(x_indx + bar_width/40 )
ax.set_xticks(x_indx + 0.022 )
ax.set_xticklabels(list(range(n_asset)), rotation=45)
ax.set_title('Percent variance explained')
ax.legend((rects[0],), ('Percent variance explained by principal components',))
plt.show();

if pca is not None:
    projected = pca.fit_transform(cov_mat)
           
# the first eigen-portfolio weights 
# first component get the Principal components
pc_w = np.zeros(num_ticker)


if pca is not None:
    pcs = pca.components_
    # pc_w = pcs[0] / pcs[0].sum(axis=0)
    pc_w = pcs[0] / pcs[0].sum(axis=0)
    
    eigen_portofilio1 = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_symbols)
    eigen_portofilio1.sort_values(by=['weights'], ascending=False, inplace=True)
    print('Sum of weights of first eigen-portfolio: %.2f' % np.sum(eigen_portofilio1))
    eigen_portofilio1.plot(title='First eigen-portfolio weights', 
                     figsize=(12,6), 
                     xticks=range(0, len(stock_symbols),10), 
                     rot=45, 
                     linewidth=3)
    plt.axhline(y=0.0, color='r', linestyle='-')
plt.show();  

fig, axes = plt.subplots(2,2, figsize=(44,22))
axesr = axes.ravel()
names = 'First Second Third Fourth Fifth Sixth'.split()

pcs = pca.components_
for idx in range(len(axesr)):
    pc_w = pcs[idx] / pcs[idx].sum(axis=0)
    name = f'{names[idx]} Eigen-portfolio weights'

    eigen_portofilio = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_symbols)
    eigen_portofilio.sort_values(by=['weights'], ascending=False, inplace=True)
    eigen_portofilio.plot(
                     xticks=range(0, len(stock_symbols),10), 
                     rot=45, 
                     linewidth=3, 
                     ax=axesr[idx])

    axesr[idx].axhline(y=0.0, color='r', linestyle='-')
    _, xmax, _, ymax = axesr[idx].axis()
    x_text = xmax * 0.95
    y_text = ymax * 0.6
    axesr[idx].text(x_text, y_text, name, ha='right', fontsize=10)
plt.subplots_adjust(left = 0.1, right = 0.95, top = 0.95, hspace = 0.3)
plt.show();



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
 
    annualized_return = ts_returns.add(1).prod() ** (periods_per_year/ts_returns.shape[0]) - 1
    annualized_vol = np.sqrt(periods_per_year*ts_returns.var())
    # annualized_vol = periods_per_year**(1/2) * ts_returns.std()
    annualized_sharpe = annualized_return / annualized_vol

    return annualized_return, annualized_vol, annualized_sharpe
           

if X_test_raw is not None:
    eigen_portofilio_returns = np.dot(X_test_raw.loc[:, eigen_portofilio1.index], eigen_portofilio1 / 100)
    eigen_portofilio_returns = pd.Series(eigen_portofilio_returns.squeeze(), index=X_norm_test.index)
    
    
    
    returns, volatility, sharpe = sharpe_ratio(eigen_portofilio_returns)
    print2('First eigen-portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (
        returns*100, volatility*100, sharpe))
    year_frac = (eigen_portofilio_returns.index[-1] - eigen_portofilio_returns.index[0]).days / 252

    df_plot = pd.DataFrame({'PC1': eigen_portofilio_returns, 'SPX': X_test_raw.loc[:, 'SPX']}, index=X_norm_test.index)
    np.cumprod(df_plot + 1).plot(title='Returns of the market-cap weighted index vs. First eigen-portfolio', 
                             figsize=(12,6), linewidth=3)  
plt.show()



pc_w = pcs[1] / pcs[1].sum(axis=0)

eigen_portofilio1 = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_symbols)
eigen_portofilio1.sort_values(by=['weights'], ascending=False, inplace=True)

if X_test_raw is not None:
    eigen_portofilio_returns = np.dot(X_test_raw.loc[:, eigen_portofilio1.index], eigen_portofilio1 / 100)
    eigen_portofilio_returns = pd.Series(eigen_portofilio_returns.squeeze(), index=X_norm_test.index)
    
    
    
    returns, volatility, sharpe = sharpe_ratio(eigen_portofilio_returns)
    print2('Second eigen-portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (
        returns*100, volatility*100, sharpe))
    year_frac = (eigen_portofilio_returns.index[-1] - eigen_portofilio_returns.index[0]).days / 252

    df_plot = pd.DataFrame({'PC1': eigen_portofilio_returns, 'SPX': X_test_raw.loc[:, 'SPX']}, index=X_norm_test.index)
    np.cumprod(df_plot + 1).plot(title='Returns of the market-cap weighted index vs. First eigen-portfolio', 
                             figsize=(12,6), linewidth=3)  
plt.show()

    
  




# set memory space for efficient and fast programming
X_norm_train, X_norm_test = None, None
X_train, X_test = None, None

# split data into train and test datasets
X_norm_train = norm_returns[norm_returns.index <= indy_].copy()
X_norm_test = norm_returns[norm_returns.index > indy_].copy()
X_train_raw = _returns[_returns.index <= indy_].copy()
X_test_raw = _returns[_returns.index > indy_].copy()

cov_mat = X_norm_train[stock_symbols].cov()
cov_matraw = X_train_raw[stock_symbols].cov()
pca = PCA()
pca.fit(cov_mat)
pcs = pca.components_


# n_portfolios = 120
n_portfolios = 283
annualized_ret = np.array([0.] * n_portfolios)
sharpe_metric = np.array([0.] * n_portfolios)
annualized_vol = np.array([0.] * n_portfolios)
idx_highest_sharpe = 0 # index into sharpe_metric which identifies a portfolio with rhe highest Sharpe ratio
    
if pca is not None:
    for ix in range(n_portfolios):
        
        ### START CODE HERE ### (≈ 4-5 lines of code)
        pc_w = pcs[:,ix] / np.sum(pcs[:, ix])
        # pc_w = pcs[ix] / pcs[ix].sum(axis=0)
        
#         print(eigen.index)
        eigen = pd.DataFrame(data ={'weights': pc_w.squeeze()}, index = stock_symbols) 
        eigen_returns = pd.Series(np.dot(X_test_raw.loc[:, eigen.index], eigen).squeeze(), index=X_norm_test.index)
        annualized_ret[ix], annualized_vol[ix], sharpe_metric[ix] = sharpe_ratio(eigen_returns)
#         print(eigen.index)
        ### END CODE HERE ###
    
    
    # find portfolio with the highest Sharpe ratio
    ### START CODE HERE ### (≈ 2-3 lines of code)
    ### ...
    
#     idx_highest_sharpe = sharpe_metric.index(max(sharpe_metric)) 
#     idx_highest_sharpe = np.argmax(sharpe_metric)

    sharpe_metric[sharpe_metric >= 100.0] = np.nan
    np.nan_to_num(sharpe_metric, nan=0, posinf=0, neginf=0, copy = False)
    idx_highest_sharpe = np.nanargmax(sharpe_metric)
    
                                       
                                       
    
    ### END CODE HERE ###
#     results = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol, 'Sharpe': sharpe_metric})
#     results.dropna(how = 'any', inplace = True)
#     results.sort_values(by=['Sharpe'], ascending=False, inplace=True)
#     print(results.head(6))

    print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' % 
          (idx_highest_sharpe,
           annualized_ret[idx_highest_sharpe]*100, 
           annualized_vol[idx_highest_sharpe]*100, 
           sharpe_metric[idx_highest_sharpe]))

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)
    ax.plot(sharpe_metric, linewidth=3)
    ax.set_title('Sharpe ratio of eigen-portfolios')
    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('Portfolios')
    plt.show()


for ix in range(n_portfolios):
    pc_w = pcs[:, ix] / np.sum(pcs[:, ix])
    eig = pd.DataFrame(data = {'weights' : pc_w.squeeze() }, index = stock_symbols)
    eig_ret = pd.Series(np.dot(X_test_raw.loc[:, eig.index], eig).squeeze(), index=X_norm_test.index)
    annualized_ret[ix], annualized_vol[ix], sharpe_metric[ix] = sharpe_ratio(eig_ret)

idx_highest_sharpe = np.nanargmax(sharpe_metric)
results = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol, 'Sharpe': sharpe_metric})
results.dropna(how = 'any', inplace = True)
results.sort_values(by=['Sharpe'], ascending=False, inplace=True)
print2(results.head(6))






annualized_ret = np.array([0.] * n_portfolios)
sharpe_metric = np.array([0.] * n_portfolios)
annualized_vol = np.array([0.] * n_portfolios)
idx_highest_sharpe = 0 # index into sharpe_metric which identifies a portfolio with rhe highest Sharpe ratio
    
if pca is not None:
    for ix in range(n_portfolios):
        
        ### START CODE HERE ### (≈ 4-5 lines of code)
        # pc_w = pcs[:,ix] / np.sum(pcs[:, ix])
        pc_w = pcs[ix] / pcs[ix].sum(axis=0)


        # pc_w = pcs[1] / pcs[1].sum(axis=0)

        eigen_portofilio1 = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_symbols)
        eigen_portofilio1.sort_values(by=['weights'], ascending=False, inplace=True)

        if X_test_raw is not None:
            eigen_portofilio_returns = np.dot(X_test_raw.loc[:, eigen_portofilio1.index], eigen_portofilio1 / 100)
            eigen_portofilio_returns = pd.Series(eigen_portofilio_returns.squeeze(), index=X_norm_test.index)



            annualized_ret[ix], annualized_vol[ix], sharpe_metric[ix] = sharpe_ratio(eigen_portofilio_returns)

    print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' % 
        (idx_highest_sharpe,
        annualized_ret[idx_highest_sharpe]*100, 
        annualized_vol[idx_highest_sharpe]*100, 
        sharpe_metric[idx_highest_sharpe]))
