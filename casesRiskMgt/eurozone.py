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
