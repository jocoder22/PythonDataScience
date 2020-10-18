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

import holoviews as hv
import hvplot.pandas

from printdescribe import print2

hv.extension('bokeh')
np.random.seed(42)

def P(*args, **kwargs):
    pt = np.linspace(-6,6,100).reshape(-1,1)
    pt = pt[pt!=0]

    return pt

def AS(pas=P(), Z_2=0):
    return pas-Z_2

def AD(pad=P(), Z_1=0):
    return -pad+Z_1

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


gdp4 = gdp.copy()
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


# Oil rents data was downloaded
oilrents = wb.download(indicator='NY.GDP.PETR.RT.ZS', country=country_list,
                start=pd.to_datetime(startdate, yearfirst=True),
                end=pd.to_datetime(enddate, yearfirst=True))\
                    .reset_index().dropna().iloc[::-1, :]\
                        .reset_index(drop=True)


# interpolate missing data
oilrents.loc[:, "NY.GDP.PETR.RT.ZS"] = oilrents.groupby('country')["NY.GDP.PETR.RT.ZS"]\
    .apply(lambda x: pd.Series(x).interpolate())


# Detrend using linear method
oilrents.loc[:, "NY.GDP.PETR.RT.ZS"] = oilrents.groupby('country')["NY.GDP.PETR.RT.ZS"]\
    .apply(lambda x: pd.Series(signal.detrend(x)))\
        .reset_index().loc[:, "NY.GDP.PETR.RT.ZS"]

# scale the oilrents against gdp
oilrents.loc[:, "NY.GDP.PETR.RT.ZS"] = oilrents.groupby('country')["NY.GDP.PETR.RT.ZS"]\
    .apply(lambda x: (x - x.iloc[0])/iqr(x))


# oilrents.loc[:, "NY.GDP.PETR.RT.ZS"] = oilrents.groupby('country')["NY.GDP.PETR.RT.ZS"]*\
#     gdp4_iqr.loc[oilrents.country].reset_index(drop=True)


# downloaded Broad Money Data
bmoney = wb.download(indicator='FM.LBL.BMNY.GD.ZS', country=country_list,
                start=pd.to_datetime(startdate, yearfirst=True),
                end=pd.to_datetime(enddate, yearfirst=True))\
                    .reset_index().dropna().iloc[::-1, :]\
                        .reset_index(drop=True)


# interpolate missing data
bmoney.loc[:, "FM.LBL.BMNY.GD.ZS"] = bmoney.groupby('country')["FM.LBL.BMNY.GD.ZS"]\
    .apply(lambda x: pd.Series(x).interpolate())


# Detrend using linear method
bmoney.loc[:, "FM.LBL.BMNY.GD.ZS"] = bmoney.groupby('country')["FM.LBL.BMNY.GD.ZS"]\
    .apply(lambda x: pd.Series(signal.detrend(x)))\
        .reset_index().loc[:, "FM.LBL.BMNY.GD.ZS"]

# scale the bmoney against gdp
bmoney.loc[:, "FM.LBL.BMNY.GD.ZS"] = bmoney.groupby('country')["FM.LBL.BMNY.GD.ZS"]\
    .apply(lambda x: (x - x.iloc[0])/iqr(x))

# bmoney.loc[:, "FM.LBL.BMNY.GD.ZS"] = bmoney.groupby('country')["FM.LBL.BMNY.GD.ZS"]*gdp4_iqr\
#     .loc[bmoney.country].reset_index(drop=True)


# converts year to numeric integers
bmoney.year = pd.to_numeric(bmoney.year)
gdp4.year = pd.to_numeric(gdp4.year)
oilrents.year = pd.to_numeric(oilrents.year)

# sanity check, to see all data align
countries = gdp4.country.unique().tolist()
years = list(reduce(np.intersect1d, [oilrents.year, 
    bmoney.year, gdp4.year]).tolist())

money = bmoney.loc[np.isin(bmoney.year, years)]
oli = oilrents.loc[np.isin(oilrents.year, years)]
gdp = gdp4.loc[np.isin(gdp4.year, years)]

print2(money)


def AS_AD(country="South Africa", year=1980):

    z_22 = oilrents.loc[((oilrents.country==country)&(oilrents.year==year)), ["NY.GDP.PETR.RT.ZS"]].fillna(0).reset_index(drop=True).iloc[0,0]
    z_11 = bmoney.loc[((oilrents.country==country)&(oilrents.year==year)), ["FM.LBL.BMNY.GD.ZS"]].fillna(0).reset_index(drop=True).iloc[0,0]

    as_eq = pd.DataFrame([P(), AS(pas=P(), Z_2=0)], index=["Price_Level", "Real Output"]).T
    ad_eq = pd.DataFrame([P(), AD(pad=P(), Z_1=0)], index=["Price_Level", "Real Output"]).T

    as_shock = pd.DataFrame([P(), AS(pas=P(), Z_2=z_22)], index=["Price_Level", "Real Output"]).T
    ad_shock = pd.DataFrame([P(), AD(pad=P(), Z_1=z_11)], index=["Price_Level", "Real Output"]).T

    result = findIntersection(lambda x: AS(pas=x, Z_2=z_22),  lambda x:AD(pad=x, Z_1=-z_11), 0.0)
    r = result + 1e-4 if result == 0 else result

    as_ad_plot = hv.Curve(as_eq, vdims="Price_Level", kdims="Real Output").options(alpha=0.2, color='#1BB3F5') *\
            hv.Curve(ad_eq, vdims="Price_Level", kdims="Real Output").options(alpha=0.2, color='orange') *\
            hv.Curve(as_shock, vdims="Price_Level", kdims="Real Output", label='AS').options(alpha=1, color='#1BB3F5') *\
            hv.Curve(ad_shock, vdims="Price_Level", kdims="Real Output", label='AD').options(alpha=1, color='orange') *\
            hv.VLine(-result[0]).options(alpha=0.2, color='black', line_width=1) *\
            hv.HLine(AS(pas=-r[0], Z_2=-z_22)).options(line_width=1, alpha=0.2, color='black')

    gdp_plot = gdp.loc[gdp.country==country].hvplot.line(y="NY.GDP.PCAP.KD", x="year") *\
        pd.DataFrame([[AS(pas=-r[0], Z_2=z_22)*0.1*gdp4_iqr_max[country], year]], columns=['GDP', 'YEAR'])\
            .hvplot.scatter(y="GDP", x='YEAR', color="red")*hv.VLine(year)
 
    return as_ad_plot+gdp_plot


# as_dict = {(a,b):AS_AD(a,b) for a in countries for b in years[1:]}
# hmap = hv.HoloMap(as_dict, kdims=['country', 'year']).collate()

# %%opts Curve [width=400, height=400]
# hmap
    
AS_AD()
# %%
