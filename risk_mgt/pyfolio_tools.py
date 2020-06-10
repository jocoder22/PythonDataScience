#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
from pandas.util.testing import assert_frame_equal
import pyfolio as pf

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

stocklist = ["C","JPM","MS", "GS"]                    
p_labels = ["Citibank", "J.P. Morgan", "Morgan Stanley", "Goldman Sachs"]

starttime = datetime.datetime(2000, 1, 1)
endtime = datetime.datetime(2019, 10, 1)

# get only the closing prices
assets = pdr.get_data_yahoo(stocklist, starttime, endtime)['Close']

