
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


def i(*args, **kwargs):
    i = np.linspace(-6,6,100).reshape(-1,1)
    i = i[i!=0]
    return i

def IS(i=i(), Z_1=0, df=True):
    return pd.DataFrame([i, -i+Z_1], index=['Real Interest','Real Output']).T if df else -i+Z_1


def LM(i=i(), Z_2=0, df=True):
    return pd.DataFrame([i, i-Z_2], index=['Real Interest','Real Output']).T if df else i-Z_2


def IP(i=i(), Z_3=0, df=True):
    return pd.DataFrame([i, i-Z_3], index=['Real Interest','Exchange Rate']).T if df else i+Z_3


def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x: fun1(x) - fun2(x), x0


