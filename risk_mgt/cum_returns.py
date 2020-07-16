# import required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
from printdescribe import print2, describe2

# initialize stock ticker
stock = "JPM"                      

# initialize the duration
starttime = datetime.datetime(2000, 1, 1)
endtime = datetime.datetime(2019, 10, 1)

# # get only the closing prices
pp = pdr.get_data_yahoo(stock, starttime, endtime)['Close']
pp.columns = ["price"]

# compute simple and log returns
# compute simple returns
pp["simple"] = pp.pct_change()

# compute log returns
pp["log"] = np.log(pp["price"]).diff()
pp["log2"] = np.log(pp["simple"] + 1)

# compute cumulative simple returns
pp["simplecum"] = (1 + pp["simple"]).cumprod() - 1

# compute cumulative log returns
pp["logcum"] =  pp["log"].cumsum()

# display the tail of data and last row of data
print2(pp.tail(), pp.iloc[-1, :])

# uses:
# simple return - portfolioreturns, short return(equal  negative(long/(long + 1)))
# log return = annualized(periodic returns), short position(which is negative long position)
# log return = foreign exchage(negative of counter currency)
