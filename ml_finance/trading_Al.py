#Import the necessary libraries
import pandas as pd
import pandas_datareader.data as dr
import matplotlib
import matplotlib.pyplot as plt
import math
import statsmodels.tsa.stattools as ts
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from printdescribe import print2

start_date = '2013-01-01'
end_date = '2020-02-29'
# end_date = '2020-09-11'

assets = 'FDN'
datasets = dr.DataReader(assets, data_source='yahoo', start = start_date)
print2(datasets.tail())


datasets['Adj Close'].plot(figsize=(10,5))
plt.show();


# generate SMA crossover trend predictions

# # compute moving averages
# fast_window = 10
# slow_window = 30

# # compute moving averages
# fast_window = 20
# slow_window = 60

# compute moving averages
fast_window = 10
slow_window = 60

# compute moving averages
# fast_window = 30
# slow_window = 60

# # compute moving averages
# fast_window = 10
# slow_window = 50

# compute moving averages
# fast_window = 20
# slow_window = 50

datasets['fast_mavg'] = datasets['Adj Close'].rolling(window=fast_window, min_periods=fast_window, center=False).mean()
datasets['slow_mavg'] = datasets['Adj Close'].rolling(window=slow_window, min_periods=slow_window, center=False).mean()

# Compute sides
datasets['side'] = np.nan

# create signal
long_signals = datasets['fast_mavg'] >= datasets['slow_mavg'] 
short_signals = datasets['fast_mavg'] < datasets['slow_mavg'] 
datasets.loc[long_signals, 'side'] = 1
datasets.loc[short_signals, 'side'] = -1

# Remove Look ahead biase by lagging the signal
datasets['side'] = datasets['side'].shift(1)
