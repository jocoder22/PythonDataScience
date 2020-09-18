
import os
import numpy as np
import pandas as pd 
import datetime
import pandas_datareader.data as dr
from printdescribe import print2, changepath


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
           'SLM', 'SPLS', 'STJ', 'SVU', 'SWY', 'TEG', 'TER', 'TGNA', 'THC', 'X', 'MAR.1', 'SPX']


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)
# pd.options.display.max_seq_items = None

patth = r"D:\PythonDataScience\ml_finance"

start_date = '2013-01-01'
# end_date = '2019-12-31'
# datasets = dr.DataReader(symbols, data_source='yahoo', start=start_date)['Adj Close']

with changepath(patth):
    # datasets.to_csv("assets.csv",  index=False,  compression='gzip')
    datasets = pd.read_csv("assets.csv",   compression='gzip')
    print2('NYT' in datasets.columns)

print2(datasets.iloc[:,191:290].info())

tt = "https://dumbstockapi.com/stock?format=tickers-only&exchange=NYSE"
pp = pd.read_json(tt)
pp = list(pp.values.ravel())

# download data and view
data2 = dr.DataReader(pp, data_source='yahoo', start=start_date)['Adj Close']
print2(f"Asset Adjusted Closing Pices shape: {data2.shape}", data2.iloc[:,10].head())

# drop columns with NaN
data2.dropna(axis=1)

# clean the datasets, remove NaN smartly
# Get a summary view of NaNs
oo = datasets.isnull().sum()

# look at the distribution through Histogram
# bimodular distribution
plt.figure(figsize=[12, 7])
oo.hist()
plt.axhline(y=50, color='r', linestyle='-')
plt.gca().set(xlabel ="Number of NaNs", ylabel="Features")
plt.show();

# Retain columns with 99.5% of data
data5 = datasets.dropna(axis=1, thresh=int(datasets.shape[0]*0.995))
print2(data5.shape, data5.isnull().sum().sum())

# Remove row with remaining NaNs
data6 = data5.dropna(axis=0)
print2(data6.shape, data6.isnull().sum().sum())

# View dataset
print2(f"Asset Adjusted Closing Pices shape: {data6.shape}", data6.iloc[:,:10].head())

# compute normalized returns
_returns = data6.pct_change()
_returns_mean = _returns.mean()
_returns_std = _returns.std()
norm_returns = (_returns - _returns_mean) / _returns_std
norm_returns.dropna(inplace=True)

# veiw normalised returns
print2(norm_returns.iloc[:, :10].head(), norm_returns.shape)



# split data into train and test datasets, based on timestamps
# 70% for the train dataset and 30% for the test dataset
index_ = data6.index[int(data6.shape[0] * 0.7)]

# set memory space for efficient and fast programming
X_norm_train, X_norm_test = None, None
X_train, X_test = None, None

X_norm_train = norm_returns[norm_returns.index <= index_].copy()
X_norm_test = norm_returns[norm_returns.index > index_].copy()
X_train = _returns[_returns.index <= index_].copy()
X_test = _returns[_returns.index > index_].copy()
