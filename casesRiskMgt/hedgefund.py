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

