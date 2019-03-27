from datetime import datetime, date
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as pdr
style.use('ggplot')

stocksname = 'TSLA'
startdate = datetime(2000, 4, 15)
enddate = date.today()

df = pdr.get_data_yahoo(stocksname, startdate, enddate)

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1)
# ax1.xaxis_date()
ax2.xaxis_date()
ax1.set_xticklabels([])

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
# plt.tight_layout()
plt.show()