
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import iqr
from scipy import signal

import pandas_datareader.wb as wb

import holoviews as hv
import hvplot.pandas

hv.extension('bokeh')
np.random.seed(42)


