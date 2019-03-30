import os
import numpy as np 
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, date
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates


from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, SimpleRNN, Dropout, Flatten
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


plt.style.use('ggplot')

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
plt.pause(2)
plt.clf()
# plt.show()

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
# closeprice = stock[['Close']].values.reshape(-1, 1)
closeprice = stock[['Close']]
closeprice = scaler.fit_transform(closeprice)

scaler2 = MinMaxScaler()
voll = scaler2.fit_transform(voll)
print(closeprice[:2], voll[:2], sep=sp, end=sp)
print(closeprice.shape, voll.shape, sep=sp, end=sp)

window = 30
val = 0.1
test = 0.1
lrate = 0.001
epoch = 10
decay_rate = lrate / epoch

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

# xdata, ydata = preprocess(voll, closeprice, window)
xdata, ydata = preprocess(closeprice, closeprice, window)
# xdata, ydata = preprocess(closeprice, closeprice, window)
xtrain, xval, xtest = train_validate_test_split2(xdata, val, test, window)
ytrain, yval, ytest = train_validate_test_split2(ydata, val, test, window)

print(xtrain.shape, xval.shape, xtest.shape, sep=sp)
print(ytrain.shape, yval.shape, ytest.shape, sep=sp)


# saving weights
savedate = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
savedir = os.path.join(os.getcwd(), 'weights')
modelname = 'Best_{0}.h5'.format(savedate)

if not os.path.isdir(savedir):
    os.makedirs(savedir)
filepath = os.path.join(savedir, modelname)

monitorbest = ModelCheckpoint(filepath=filepath, monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

callbacks=[monitorbest]

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(window, 1)))
model.add(Dropout(0.3))

model.add(LSTM(190))
model.add(Dropout(0.3))

model.add(Dense(5))

model.add(Dense(1))

adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
model.compile(optimizer='adam', loss='mse')  


history = model.fit(xtrain, ytrain, epochs=epoch, validation_data=(xval, yval), 
            callbacks=callbacks, shuffle=False)

# to load only the weights you must define the model as above

# model.load_weights('weights\Best_2019_03_28 02_07_59.h5') # for both vol and closing price
# model.fit with callbacks save only the weights


# model = load_model(f"model\model55.h5")
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='Val_loss')
plt.legend()
plt.pause(2)
plt.clf()
plt.show()

print(model.summary())
pred = model.predict(xtest)
print(type(pred), ytest.shape, sep=sp, end=sp)
# actual, prediction = [], []
prediction = scaler.inverse_transform(pred).reshape(-1)
actual= scaler.inverse_transform(ytest).reshape(-1)
df = pd.DataFrame({'Actual': actual, 'Predictions': prediction})

print(df.shape, df.head(), df.tail(), sep=sp, end=sp)
plt.plot(df.Actual.values, label='Actual', color='red')
plt.plot(df.Predictions.values, label='Prediction', color='yellow')
plt.legend()
plt.show()

# now = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
# save model
# model.save_weights('model_lstm.h5')
# model.save(f"model\model55.h5")
