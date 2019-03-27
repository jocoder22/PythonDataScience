#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, date
from mpl_finance import candlestick2_ohlc
from matplotlib.dates import MonthLocator, date2num, DateFormatter,  WeekdayLocator,\
    DayLocator, MONDAY
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# https://www.lfd.uci.edu/~gohlke/pythonlibs/
# http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
# http://codetheory.in/how-to-convert-a-video-with-python-and-ffmpeg/


stocksname = 'LNG'
startdate = datetime(2000, 4, 15)
enddate = date.today()

stock = pdr.get_data_yahoo(stocksname, startdate, enddate)
stock = stock['2017':]
stock.reset_index(inplace=True)
stock['Date'] = stock['Date'].map(mdates.date2num)

print(stock.head())

mondays = WeekdayLocator(MONDAY)        
alldays = DayLocator()              
weekFormatter = DateFormatter('%b %d')  
dayFormatter = DateFormatter('%d') 

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
ax.xaxis_date()

candlestick2_ohlc(ax,
                  opens=stock.Open,
                  closes=stock.Close,
                  highs=stock.High,
                  lows=stock.Low,
                  width=0.5,
                  colordown='red',
                  colorup='green')

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.legend()
ax.xaxis_date()
plt.show()


stock.Close.plot()
plt.show()



mondays = WeekdayLocator(MONDAY)        
alldays = DayLocator()              
weekFormatter = DateFormatter('%b %d')  
dayFormatter = DateFormatter('%d')      

ax.xaxis.set_major_locator(mticker.MaxNLocator(10))

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
ax.xaxis.set_minor_formatter(dayFormatter)

candlestick2_ohlc(ax,
                  opens=stock.Open,
                  closes=stock.Close,
                  highs=stock.High,
                  lows=stock.Low,
                  width=0.5,
                  colordown='red',
                  colorup='green')

ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()