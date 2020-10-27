#!/usr/bin/env python
import os
import pickle
from functools import reduce
from operator import mul

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from sklearn import linear_model
from sklearn.decomposition import PCA

import holoviews as hv
import hvplot
import hvplot.pandas

np.random.seed(42)
hv.extension('bokeh')

pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web

# Download datasets
portfolios100 = web.DataReader('100_Portfolios_10x10_Daily', 'famafrench')
factors5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_Daily', 'famafrench')

print(portfolios100['DESCR'])
print(factors5['DESCR'])

# select the Average Equal Weighted Returns -- Daily (1220 rows x 100 cols)
portfolios = portfolios100[1]
factors = factors5[0]

# pd.melt(portfolios.head(100).divide(100).add(1).cumprod().reset_index(), 
#         id_vars='Date').plot.line(x='Date', by='variable')
pp = portfolios.iloc[:,0:12].divide(100).add(1).cumprod()
pp.plot()
plt.show()