
import datetime
import pandas_datareader as pdr

symbol = 'AAPL'
starttime = datetime.datetime(2015, 1, 1)
endtime = datetime.datetime(2018, 12, 31)
apple = pdr.get_data_yahoo(symbol, starttime, endtime)
type(apple)
# <class 'pandas.core.frame.DataFrame'>
apple.to_csv('apple.csv')
