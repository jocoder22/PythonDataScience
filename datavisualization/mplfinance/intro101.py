#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, date
from mpl_finance import candlestick2_ohlc


# https://www.lfd.uci.edu/~gohlke/pythonlibs/
# http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
# http://codetheory.in/how-to-convert-a-video-with-python-and-ffmpeg/


stocksname = ['LNG']
startdate = datetime(2000, 4, 15)
enddate = date.today()

lng_df = pdr.get_data_yahoo(stocksname, startdate, enddate)


