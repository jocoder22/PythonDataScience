#!/usr/bin/env python
# Import required modules
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr


# My first stock is Neflix
# I'm using daily close prices
stock_1 = "NFLX"

# using 2 years of data from January 01, 2018 to December 31, 2019
starttime = datetime.datetime(2018, 1, 1)
endtime = datetime.datetime(2019, 12, 31)

# get only the closing prices
neflix = pdr.get_data_yahoo(stock_1, starttime, endtime)['Close']


# Calculate log return
logReturn_neflix = np.log(neflix).diff().dropna()
logReturn_neflix.head()