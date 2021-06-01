#!/usr/bin/env python
import pandas as pd
import matplotlib.ticker as mticker
import mplfinance as mpf
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates
import finplot as fplt
import pandas_datareader as pdr
from datetime import datetime, date

# aapl = yf.download('BABA', '2020-1-1','2021-2-18')


symbol = 'BABA' # 'AAPL' #'AMZN'  
starttime = datetime(2021, 1, 1)
endtime = date.today()
aapl = pdr.get_data_yahoo(symbol, starttime, endtime)
print(aapl.head())

# mpf.plot(aapl)
mpf.plot(aapl, type='candle')
mpf.plot(aapl, type='candle', mav=(12,26,9))

fplt.background = '#B0E0E6'
fplt.candlestick_ochl(aapl[['Open', 'Close', 'High', 'Low']])
fplt.show()

fplt.background = '#F5F5F5'
fplt.candlestick_ochl(aapl[['Open', 'Close', 'High', 'Low']])
fplt.show()


fplt.background = "#BDB76B"
fplt.odd_plot_background = '#f0f' # purple
fplt.plot(aapl.Close)
fplt.show()

fplt.background = "#B0C4DE"
fplt.candlestick_ochl(aapl[['Open', 'Close', 'High', 'Low']])
fplt.show()


ax, ax2 = fplt.create_plot('Apple MACD', rows=2)
fplt.background = "#fff"
# plot macd with standard colors first
macd = aapl.Close.ewm(span=12).mean() - aapl.Close.ewm(span=26).mean()
signal = macd.ewm(span=9).mean()

aapl['macd_diff'] = macd - signal
fplt.volume_ocv(aapl[['Open','Close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
fplt.plot(macd, ax=ax2, legend='MACD')
fplt.plot(signal, ax=ax2, legend='Signal')


# change to b/w coloring templates for next plots
fplt.candle_bull_color = fplt.candle_bear_color = '#000'
fplt.volume_bull_color = fplt.volume_bear_color = '#333'
fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#fff'

# plot price and volume
fplt.candlestick_ochl(aapl[['Open','Close','High','Low']], ax=ax)
hover_label = fplt.add_legend('', ax=ax)
axo = ax.overlay()
fplt.volume_ocv(aapl[['Open','Close','Volume']], ax=axo)
fplt.plot(aapl.Volume.ewm(span=24).mean(), ax=axo, color=1)

fplt.show()


# https://pypi.org/project/finplot/