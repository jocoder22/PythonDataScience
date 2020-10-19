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

indicators = wb.get_indicators()
indicators.shape

indicators.source.value_counts()

wbb = indicators[indicators.source == 'World Development Indicators']
wbb.shape
wbb.topics.value_counts()


wbb_eg_ext_debt = wbb[wbb.topics == "Economy & Growth ; External Debt"]
wbb_eg_ext_debt

wbb_eg_ext_debt.iloc[:,[0,1,3]]



