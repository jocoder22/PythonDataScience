#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, date
from mpl_finance import candlestick2_ohlc
from matplotlib.dates import MonthLocator, date2num, DateFormatter


# https://www.lfd.uci.edu/~gohlke/pythonlibs/
# http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
# http://codetheory.in/how-to-convert-a-video-with-python-and-ffmpeg/


stocksname = 'LNG'
startdate = datetime(2000, 4, 15)
enddate = date.today()

stock = pdr.get_data_yahoo(stocksname, startdate, enddate)
stock = stock['2017':]

print(stock.head())

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.5)

loc = MonthLocator()
fmt = DateFormatter('%b')

ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(fmt)

candlestick2_ohlc(ax,
                  opens=stock.Open,
                  closes=stock.Close,
                  highs=stock.High,
                  lows=stock.Low,
                  width=0.3,
                  colordown='red',
                  colorup='green')

plt.tight_layout()
plt.legend()
plt.show()


stock.Close.plot()
plt.show()