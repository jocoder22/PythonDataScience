#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, date
import plotly.plotly as py
import plotly.graph_objs as go

path = r'C:\Users\Jose\Desktop\PythonDataScience\datavisualization\plotly'
os.chdir(path)

stocksname = 'AAPL'
startdate = datetime(2000, 4, 15)
enddate = date.today()

stock = pdr.get_data_yahoo(stocksname, startdate, enddate)

stock.reset_index(inplace=True)
print(stock.head())

trace = go.Ohlc(x=stock['Date'],
                open=stock['Open'],
                high=stock['High'],
                low=stock['Low'],
                close=stock['Close'])
data = [trace]
py.iplot(data, filename='simple_candlestick')