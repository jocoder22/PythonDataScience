#!/usr/bin/env python
import numpy as np
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
sp = {"end":"\n\n", "sep":"\n\n"}

symbol = "VZ" # "CMCSA" #"VZ" #'BABA' # 'AAPL' #'AMZN'  
starttime = datetime(2021, 1, 1)
endtime = date.today()
aapl = pdr.get_data_yahoo(symbol, starttime, endtime)
print(aapl.head(), aapl.tail(), **sp)


d = [12,26,9]
w = [5,35,5]

dayy = False
if dayy == True:
    dd = d
else: dd = w

# computer MACD and signal
macd = aapl.Close.ewm(span=dd[0]).mean() - aapl.Close.ewm(span=dd[1]).mean()
signal = macd.ewm(span=dd[2]).mean()
aapl['macd_diff'] = macd - signal

# form dataframe
aapl['MACD'] = macd 
aapl['MACDsig'] = signal

# compute period mean volume
aapl['numb'] = np.arange(1, aapl.shape[0]+1)
aapl['CUMSUM_C'] = aapl['Volume'].cumsum()
aapl["aveg"] = aapl['CUMSUM_C']/aapl['numb'] 


print(aapl.head(), aapl.tail(), **sp)


fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize =(24,10))
ax1.grid(alpha=0.7); ax2.grid(alpha=0.7)
ax1.plot(aapl.aveg, color="black")
ax2.plot(aapl.Volume, color="red")
plt.show()


fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize =(24,10))
ax1.grid(alpha=0.7); ax2.grid(alpha=0.7)
ax1.set_title("Candlestick"); ax2.set_title("MACD")



color = ["green" if close_price > open_price else "red" for close_price, open_price in zip(aapl.Close, aapl.Open)]
ax1.bar(x=aapl.index, height=np.abs(aapl.Open-aapl.Close), bottom=np.min((aapl.Open,aapl.Close), axis=0), width=0.6, color=color)
ax1.bar(x=aapl.index, height=aapl.High - aapl.Low, bottom=aapl.Low, width=0.1, color=color)

ax3 = ax2.twinx()
ax3.plot(aapl.Volume, color="black")

# plt.title(f'MACD chart {symbol}')
color2 = ["green" if close_price > open_price else "red" for close_price, open_price in zip(aapl.MACD, aapl.MACDsig)]
ax2.plot( aapl['MACD'], label='MACD')
ax2.plot( aapl['MACDsig'], label='MACDsig')
# ax2.plot( aapl['macd_diff'],  label='MACDhist')
ax2.bar( aapl.index, aapl['macd_diff'], snap=False, color = color2, width=0.6, label='MACDhist')
ax2.legend()

plt.show()


def candlestick(t, o, h, l, c):
    plt.figure(figsize=(12,4))
    color = ["green" if close_price > open_price else "red" for close_price, open_price in zip(c, o)]
    plt.bar(x=t, height=np.abs(o-c), bottom=np.min((o,c), axis=0), width=0.6, color=color)
    plt.bar(x=t, height=h-l, bottom=l, width=0.1, color=color)

candlestick(
    aapl.index,
    aapl.Open,
   aapl.High,
   aapl.Low,
   aapl.Close
)

plt.grid(alpha=0.9)
plt.show()

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



fplt.background = "#fff"
ax, ax2 = fplt.create_plot('Apple MACD', rows=2)
fplt.background = "#fff"

# plot macd with standard colors first
fplt.background = "#fff"
fplt.volume_ocv(aapl[['Open','Close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
fplt.background = "#fff"
fplt.plot(macd, ax=ax2, legend='MACD')
fplt.plot(signal, ax=ax2, legend='Signal')

# change to b/w coloring templates for next plots
fplt.candle_bull_color = fplt.candle_bear_color = '#000'
fplt.volume_bull_color = fplt.volume_bear_color = '#333'
fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#fff'

# plot price and volume
fplt.background = "#fff"
fplt.candlestick_ochl(aapl[['Open','Close','High','Low']], ax=ax)
hover_label = fplt.add_legend('', ax=ax)
axo = ax.overlay()
fplt.volume_ocv(aapl[['Open','Close','Volume']], ax=axo)
fplt.plot(aapl.Volume.ewm(span=24).mean(), ax=axo, color=1)

fplt.show()


# https://pypi.org/project/finplot/

aapl['MACD'] = macd 
aapl['MACDsig'] = signal
plt.title(f'MACD chart {symbol}')
plt.plot( aapl['MACD'].fillna(0), label='MACD')
plt.plot( aapl['MACDsig'].fillna(0), label='MACDsig')
plt.plot( aapl['macd_diff'].fillna(0), label='MACDhist')
plt.bar( aapl.index, aapl['macd_diff'].fillna(0), width=0.1, snap=False, label='MACDhist')
plt.legend()

plt.show()