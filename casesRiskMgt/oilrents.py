#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import iqr
from scipy import signal

import statsmodels.api as sm

import pandas_datareader.wb as wb

import holoviews as hv
import hvplot.pandas

from printdescribe import print2

hv.extension('bokeh')
np.random.seed(42)

def P(*args, **kwargs):
    p = np.linspace(-6,6,100).reshape(-1,1)
    p = p[p!=0]

    return p

def AS(P=P(), Z_2=0):
    return p-Z_2

def AD(P=P(), Z_1=0):
    return -p+Z_1

def findIntersection(fun1,fun2, x0):
    return fsolve(lambda x: fun1(x) - fun2(x), x0)


country_list = ['USA', 'GBR', 'MEX', 'CAN', 'ZAF', 'NGA']
startdate = '1970'
enddate = '2019'
crisis_year = pd.to_datetime('1987-01-01')

gdp = wb.download(indicator='NY.GDP.PCAP.KD', country=country_list,
                start=pd.to_datetime(startdate, yearfirst=True),
                end=pd.to_datetime(enddate, yearfirst=True))\
                    .reset_index().dropna().iloc[::-1, :]


print2(gdp.shape, gdp.head(), gdp.info(), gdp.country.value_counts(dropna=False))


gdp2 = gdp.copy()
gdp2['year'] = pd.to_datetime(gdp2['year'])
gdp2.set_index('year', inplace=True)
gdp2.loc[:, "NY.GDP.PCAP.KD"] = gdp2.groupby('country')["NY.GDP.PCAP.KD"]\
    .apply(lambda x: pd.Series(x).interpolate())

gdp2.groupby(['country'])['NY.GDP.PCAP.KD'].plot()
plt.axvline(crisis_year, color="black")
plt.legend()
plt.show()

print2(gdp2.info())




gdp3 = gdp.copy()
gdp3['year'] = pd.to_datetime(gdp3['year'])
gdp3.loc[:, "NY.GDP.PCAP.KD"] = gdp3.groupby('country')["NY.GDP.PCAP.KD"]\
    .apply(lambda x: pd.Series(signal.detrend(x)))\
        .reset_index().loc[:, "NY.GDP.PCAP.KD"]
gdp3.set_index('year', inplace=True)
maxusa = gdp3.groupby('country')["NY.GDP.PCAP.KD"].idxmax()["United States"]

gdp3.groupby(['country'])['NY.GDP.PCAP.KD'].plot()
plt.axvline(crisis_year, color="black")
plt.axvline(maxusa, color="black")
plt.legend()
plt.show()

print2(gdp3.groupby('country')["NY.GDP.PCAP.KD"].idxmax()["United States"])

maxusa = gdp3.groupby('country')["NY.GDP.PCAP.KD"].idxmax()["United States"]


gdp33 = gdp.copy()
gdp33['year'] = pd.to_datetime(gdp33['year'])
gdp33.loc[:, "NY.GDP.PCAP.KD"] = gdp33.groupby('country')["NY.GDP.PCAP.KD"]\
    .apply(lambda x: pd.Series(x).diff())\
        .reset_index().loc[:, "NY.GDP.PCAP.KD"]
gdp33.set_index('year', inplace=True)
maxusa = gdp3.groupby('country')["NY.GDP.PCAP.KD"].idxmax()["United States"]

gdp33.groupby(['country'])['NY.GDP.PCAP.KD'].plot()
plt.axvline(crisis_year, color="black")
plt.axvline(maxusa, color="black")
plt.legend()
plt.show()


# plot the graphs
fig, ax = plt.subplots(nrows=3, sharex=True)
gdp2.groupby(['country'])['NY.GDP.PCAP.KD'].plot(ax=ax[0])
gdp3.groupby(['country'])['NY.GDP.PCAP.KD'].plot(ax=ax[1])
gdp33.groupby(['country'])['NY.GDP.PCAP.KD'].plot(ax=ax[2])
for i in range(0,3):
    ax[i].axvline(crisis_year, color="black")
    ax[i].axvline(maxusa, color="black")
plt.legend()
plt.show()


gdp4 = gdp()
# interpolate missing data
gdp4.loc[:, "NY.GDP.PCAP.KD"] = gdp4.groupby('country')["NY.GDP.PCAP.KD"]\
    .apply(lambda x: pd.Series(x).interpolate())


# Detrend using linear method
gdp4.loc[:, "NY.GDP.PCAP.KD"] = gdp4.groupby('country')["NY.GDP.PCAP.KD"]\
    .apply(lambda x: pd.Series(signal.detrend(x)))\
        .reset_index().loc[:, "NY.GDP.PCAP.KD"]

# scale the dataset
gdp4.loc[:, "NY.GDP.PCAP.KD"] = gdp4.groupby('country')["NY.GDP.PCAP.KD"]\
    .apply(lambda x: (x - x.iloc[0])/iqr(x))

gdp4_iqr = gdp4.groupby('country')["NY.GDP.PCAP.KD"]\
    .apply(lambda x: iqr(x))
gdp4_iqr_max = gdp4.groupby('country')["NY.GDP.PCAP.KD"]\
    .apply(lambda x: iqr(x, rng=(1,99)))
        
