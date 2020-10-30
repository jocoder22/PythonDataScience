import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import date, datetime

startdate = datetime(2000, 1, 1)
enddate = date.today()

tickers = 'FB AMZN AAPL GOOGL NFLX MSFT ^GSPC'.split()
portfolio = pdr.get_data_yahoo(tickers, startdate, enddate)['Adj Close']
