#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import iqr
from scipy import signal

import statsmodels.api as sm
from functools import reduce
import operator

import pandas_datareader.wb as wb
import wbdata

import holoviews as hv
import hvplot.pandas

from printdescribe import print2

hv.extension('bokeh')
np.random.seed(42)

indicators = wb.get_indicators()
indicators.shape

indicators.source.value_counts()

wbb = indicators[indicators.source == 'World Development Indicators']
wbb.shape
wbb.topics.value_counts()


wbb_eg_ext_debt = wbb[wbb.topics == "Economy & Growth ; External Debt"]
wbb_eg_ext_debt

wbb_eg_ext_debt.iloc[:,[0,1,3]]


countries = 'ESP IRL'.split()
indicators = {'GC.DOD.TOTL.GD.ZS':'Debt total'}
 
#grab indicators above for countires above and load into data frame
df = wbdata.get_dataframe(indicators, country=countries, convert_date=False)

#df is "pivoted", pandas' unstack fucntion helps reshape it into something plottable
dfu = df.unstack(level=0).dropna()

# a simple matplotlib plot with legend, labels and a title
plt.rcParams["figure.figsize"] = (10,6)
dfu.plot(); 
plt.legend(loc='best'); 
plt.title("Total Central government debt (Percentage of GDP)"); 
plt.xlabel('Date'); plt.ylabel('Total debt (% of GDP)');






indicator1 = {'CM.MKT.INDX.ZG':'External debt'}
countries = 'GRC ESP PRT IRL'.split()

# #grab indicators above for countires above and load into data frame
# dat = wb.download(indicator='DT.DOD.DECT.GN.ZS', country=countries,  start=1960, end=2020)
# # DT.DOD.DECT.GN.ZS	External debt stocks (% of GNI)

#df2['date']=pd.to_numeric(df2['date'])
# df_selyrs=df2[(df2['date']>=2014) & (df2['date']<=2016)]


#grab indicators above for countires above and load into data frame
# df = wbdata.get_dataframe(indicators, country=countries, convert_date=True)
# df = df[df.date >= 2005 & df.date <= 2012]
# df.reset_index(inplace=True)
# grab indicators above for countires above and load into data frame
df = wb.download(indicator=indicator1, country=countries,  start=2005, end=2012)

# df is "pivoted", pandas' unstack fucntion helps reshape it into something plottable
df = df.unstack(level=0).dropna()
# df.reset_index(inplace=True)
# dfu = df[df['date'] >= 2005 & df['date'] <=2013]

# a simple matplotlib plot with legend, labels and a title
plt.rcParams["figure.figsize"] = (10,6)
dfu.plot(); 
plt.legend(loc='best'); 
plt.title("Central government debt, total (% of GDP)"); 
plt.xlabel('Date'); plt.ylabel('Total debt (% of GDP)');





# CM.MKT.LCAP.CD
indicator2 = {'CM.MKT.LCAP.CD':'External debt'}

df = wb.download(indicator=indicator2, country=countries,  start=2005, end=2012)

# df is "pivoted", pandas' unstack fucntion helps reshape it into something plottable
df = df.unstack(level=0).dropna()

# a simple matplotlib plot with legend, labels and a title
plt.rcParams["figure.figsize"] = (10,6)
dfu.plot(); 
plt.legend(loc='best'); 
plt.title("Central government debt, total (% of GDP)"); 
plt.xlabel('Date'); plt.ylabel('Total debt (% of GDP)');




df.loc[(df.index.get_level_values('date') >= '2016-01-01') &
       (df.index.get_level_values('date') <= '2019-01-01')]

# CM.MKT.LCAP.CD
indicator = {'CM.MKT.LCAP.CD':'External debt'}
countries = 'GRC ESP PRT IRL'.split()

# df = wb.download(indicator=indicator2, country=countries,  start=2005, end=2012)
#grab indicators above for countires above and load into data frame
df = wbdata.get_dataframe(indicator, country=countries, convert_date=True)

idx = pd.IndexSlice
df.loc[idx[:, ['2010-01-01','2017-01-01']], idx[:]].head()

df.loc[(df.index.get_level_values('date') >= '2016-01-01') &
       (df.index.get_level_values('date') <= '2019-01-01')]

df = wbdata.get_dataframe(indicator, country=countries, convert_date=True)
query = df.index.get_level_values(1) >= pd.Timestamp('2013-01-01')
query2 = df2.index.get_level_values(1) <= pd.Timestamp('2018-01-01')
df2 = df[query]
df2[query2]
