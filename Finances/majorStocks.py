import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime

startdate = datetime(1996, 1, 1)
enddate = date.today()

ticker = 'FG AMZN AAPL GOOGL NFLX MSFT ^GSPC'.split()
portfolio = pdr.get_data_yahoo(symbol, startdate, enddate)['Adj Close']
