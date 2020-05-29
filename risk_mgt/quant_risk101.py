import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
from pandas.util.testing import assert_frame_equal


stocklist = ["C","JPM","MS", "GS"]

starttime = datetime.datetime(2000, 1, 1)
apple = pdr.get_data_yahoo(stocklist, starttime)

print(apple.head(), apple.columns, apple.info())