#!/usr/bin/env python
import pandas as pd
import matplotlib.ticker as mticker
import mplfinance as mpf
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates
import finplot as fplt

aapl = yf.download('AAPL', '2020-1-1','2021-2-18')
print(aapl.head())

# mpf.plot(aapl)
mpf.plot(aapl, type='candle')
mpf.plot(aapl, type='candle',mav=(12,26,9))

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
