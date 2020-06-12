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

stocklist = ["C","JPM","MS", "GS", "^GSPC"]                    
p_labels = ["Citibank", "J.P. Morgan", "Morgan Stanley", "Goldman Sachs", "SP500"]

starttime = datetime.datetime(2000, 1, 1)
endtime = datetime.datetime(2020, 5, 1)

# get only the closing prices
assets = pdr.get_data_yahoo(stocklist, starttime, endtime)['Close']

returns = assets.pct_change()

returns.index = pd.to_datetime(returns.index)

# Create tear sheat
fig = returns.create_returns_tear_sheat(returns, return_fig=True)

# using backtest and live data
rts_live = returns.create_returns_tear_sheat(returns, start_live_date="2019-03-15")

def display_tear_sheet():
  p = 'pyfolio_tear_sheet_3.13.png'
  HtmlManager.getPlots().append(FigHtml([p]))

 display_tear_sheet()
