#!/usr/bin/env python
## https://pandas-datareader.readthedocs.io/en/latest/index.html

import datetime
import pandas_datareader as pdr
from pandas_datareader import wb  # for world bank datasets

symbol = 'AAPL'
starttime = datetime.datetime(2015, 1, 1)
endtime = datetime.datetime(2018, 12, 31)
apple = pdr.get_data_yahoo(symbol, starttime, endtime)
type(apple)
# <class 'pandas.core.frame.DataFrame'>
apple.to_csv('apple.csv')

symbol2 = '^GSPC'
starttime2 = datetime.datetime(2010, 1, 1)
endtime2 = datetime.datetime(2019, 1, 7)
sp500 = pdr.get_data_yahoo(symbol2, starttime2, endtime2)
sp500.to_csv('sp500.csv')




# for world bank dataset
 matches = wb.search('gdp.*capita.*const')
 data = wb.download(indicator='NY.GDP.PCAP.KD', 
                   country=['US', 'CA', 'MX'], 
                   start=2005, end=2008)


data.head()
data['NY.GDP.PCAP.KD'].groupby(level=0).mean()
wb.search('cell.*%').iloc[:, :2]
ind = ['NY.GDP.PCAP.KD', 'IT.MOB.COV.ZS']

ind = ['NY.GDP.PCAP.KD', 'IT.MOB.COV.ZS']
Idat = wb.download(indicator=ind, country='all',
                          start=2011, end=2011).dropna()
Idat.columns = ['gdp', 'cellphone']
print(Idat.head())
print(Idat.tail())


# import sys
# python - m pip install - -upgrade pip
# !{sys.executable} - m pip install pandas-datareader
