#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from datetime import datetime, date
import seaborn as sns
# Import the statsmodels.api library with the alias sm
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_pacf

# https://www.lfd.uci.edu/~gohlke/pythonlibs/
# http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
# http://codetheory.in/how-to-convert-a-video-with-python-and-ffmpeg/
import talib

stocksname = ['LNG']
startdate = datetime(2000, 4, 15)
enddate = date.today()

lng_df = pdr.get_data_yahoo(stocksname, startdate, enddate)