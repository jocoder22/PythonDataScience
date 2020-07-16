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
pp["simple"] = pp.pct_change()*100
pp["log"] = np.log(pp["price"]).diff()*100
pp["simplecum"] = (1 + pp["simple"]).cumprod() - 1
