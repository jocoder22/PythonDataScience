#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
import statsmodels.api as sm
from pandas.util.testing import assert_frame_equal

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

stocklist = ["C","JPM","MS", "GS"]
stocklist = ["JPM", "GS", "BAC", "MS", "C","CS",
             "BCS" , "DB", "UBS", "RY", "WFC",
             "HSBC", "JFE", "BNP.PA", "MFG", "LAZ", "NMR", "EVR",
             "BMO", "MUFG"]
             
             
p_labels = ["Citibank", "J.P. Morgan", "Morgan Stanley", "Goldman Sachs"]
pp_labels = ["JPMorgan Chase", "Goldman Sachs", "BofA Securities", "Morgan Stanley", "Citigroup", "Credit Suisse", 
             "Barclays Investment Bank", "Deutsche Bank", "UBS", "RBC Capital Markets", "Wells Fargo Securities",
             "HSBC", "Jefferies Group", "BNP Paribas", "Mizuho", "Lazard", "Nomura", "Evercore Partners", 
             "BMO Capital Markets", "Mitsubishi UFJ Financial Group"]

starttime = datetime.datetime(2000, 1, 1)
endtime = datetime.datetime(2019, 10, 1)

# get only the closing prices
portfolio = pdr.get_data_yahoo(stocklist, starttime, endtime)['Close']
