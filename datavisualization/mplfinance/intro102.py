from datetime import datetime, date
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as pdr

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, SimpleRNN, Dropout, Flatten
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


style.use('ggplot')

stocksname = 'TSLA'
startdate = datetime(1990, 4, 15)
enddate = date.today()

df = pdr.get_data_yahoo(stocksname, startdate, enddate)

# df_ohlc = df['Adj Close'].resample('10D').ohlc()
# df_volume = df['Volume'].resample('10D').sum()

df_ohlc = df['High'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

ax1 = plt.subplot2grid((7,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((7,1), (5,0), rowspan=2, colspan=1)
# ax1.xaxis_date()
ax2.xaxis_date()
ax1.set_xticklabels([])

candlestick_ohlc(ax1, df_ohlc.values, width=4, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.tight_layout()
plt.show()

# https://pythonprogramming.net/sp500-company-price-data-python-programming-for-finance/?completed=/sp500-company-list-python-programming-for-finance/

path = r'C:\Users\Jose\Desktop\PythonDataScience\RNN'
os.chdir(path)

sp = '\n\n'
symbol = 'AAL' # 'AAPL' #'AMZN'  
starttime = datetime(1996, 1, 1)
endtime = date.today()
stock = pdr.get_data_yahoo(symbol, starttime, endtime)
stock.reset_index(inplace=True)
print(stock.head(), stock.shape, sep=sp)

scaler = MinMaxScaler()
voll = stock[['Volume']]
closeprice = stock[['Close']]
closeprice = scaler.fit_transform(closeprice)
voll = scaler.fit_transform(voll)
print(closeprice, voll, sep=sp)

window = 14
val = 0.1
test = 0.1

def preprocess(data, data2, wdw):
    feature, target = [], []
    for idx in range(len(data) - wdw - 1):
        feature.append(data[idx: idx + wdw, 0])
        target.append(data2[idx + wdw, 0])

    return np.array(feature), np.array(target)


def train_validate_test_split2(datatt, tx, vx, ww):
    vxx = tx + vx
    test, validate, train = np.split(datatt, [int(tx*len(datatt)), int(vxx*len(datatt))])
    return np.expand_dims(train, axis=-1), np.expand_dims(validate, axis=-1), np.expand_dims(test, axis=-1)

xdata, ydata = preprocess(closeprice, voll, window)

xtrain, xval, xtest = train_validate_test_split2(xdata, val, test, window)
ytrain, yval, ytest = train_validate_test_split2(ydata, val, test, window)

print(xtrain.shape, xval.shape, xtest.shape, sep=sp)
print(ytrain.shape, yval.shape, ytest.shape, sep=sp)