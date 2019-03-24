#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import sklearn
from datetime import datetime, date
import tensorflow as tf

import talib as tb
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, Dropout, Flatten
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


import pandas_datareader as pdr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



sp = '\n\n'
symbol = 'AAL'
# symbol = 'RELIANCE.NS'
starttime = datetime(1996, 1, 1)
endtime = date.today()
ALL = pdr.get_data_yahoo(symbol, starttime, endtime)[['Close']]
ALL['maxClose'] = ALL.Close.max()
ALL['minClose'] = ALL.Close.min()
ALL['%Change'] = ALL.Close.pct_change()
print(ALL.head())



