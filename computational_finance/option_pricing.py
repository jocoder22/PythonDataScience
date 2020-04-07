#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, date, timedelta

from iexfinance.refdata import get_symbols
from iexfinance.stocks import Stock, get_historical_intraday, get_historical_data

pathtk = r"D:\PPP"
sys.path.insert(0, pathtk)

import wewebs


def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")


sp = {'sep': '\n\n', 'end': '\n\n'}

path = r"D:\Intradays"

ttt = wewebs.token

stock = "NFLX"

startdate = datetime(2016, 2, 2)
enddate = datetime(2018, 5, 30)
stdate = date.today() - timedelta(days=456)


# allstocks = pdr.get_data_yahoo(stock, startdate)['Adj Close']
# print(allstocks.head())
get_symbols(output_format='pandas', token=ttt)

neflex = Stock(stock, token=ttt)
print2(neflex.get_quote()['close'])

start = datetime(2017, 1, 1)
end = datetime(2018, 1, 1)

df = get_historical_data("TSLA", start, end, token=ttt, output_format='pandas')

print2(df.close.var())
