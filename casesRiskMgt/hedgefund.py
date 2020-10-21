#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce
from operator import mul

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import linear_model
from sklearn.decomposition import PCA

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.wb as wb

import holoviews as hv
import hvplot
import hvplot.pandas


from printdescribe import print2

hv.extension('bokeh')
np.random.seed(42)

