#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")


stocklist = ["JPM", "GS", "BAC", "MS", "C","CS"]                         
pp_labels = ["JPMorgan Chase", "Goldman Sachs", "BofA Securities", "Morgan Stanley", "Citigroup", "Credit Suisse"] 

starttime = datetime.datetime(2000, 1, 1)
endtime = datetime.datetime(2019, 10, 1)

# get only the closing prices
assets = pdr.get_data_yahoo(stocklist, starttime, endtime)['Close']