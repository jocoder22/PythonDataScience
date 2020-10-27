#!/usr/bin/env python
import os
import pickle
from functools import reduce
from operator import mul

import pandas as pd
import numpy as np

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