import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import date, datetime

startdate = datetime(2000, 1, 1)
enddate = date.today()

tickers = 'FB AMZN AAPL GOOGL NFLX MSFT ^GSPC'.split()
portfolio = pdr.get_data_yahoo(tickers, startdate, enddate)['Adj Close']

portfolio.iloc[:,[0,2,4]].plot(figsize=(14,8))
plt.grid(color='black', which='major', axis='y', linestyle='solid')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3);
