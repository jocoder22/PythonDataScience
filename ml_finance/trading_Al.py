#Import the necessary libraries
import pandas as pd
import pandas_datareader.data as dr
import matplotlib
import matplotlib.pyplot as plt
import math
import statsmodels.tsa.stattools as ts
import numpy as np
from scipy import stats


import mlfinlab as ml

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


datasets['fast_mavg'].dropna().plot(figsize=(10,5))
datasets['slow_mavg'].dropna().plot()
(datasets['side']*200).dropna().plot()
plt.legend();


datasets['side2'] = np.where(datasets.side > 0, datasets.side *200,0)
datasets['fast_mavg'].dropna().plot(figsize=(10,5))
datasets['slow_mavg'].dropna().plot()
(datasets['side2']).dropna().plot()
plt.legend();



orig_data = datasets.copy()

# Drop the NaN values from our data set
datasets.dropna(axis=0, how='any', inplace=True)

print("1.0 --> Long Signals")
print("-1.0 --> Short Signals")
print("--------------------")
print(datasets['side'].value_counts())




# Compute daily volatility
daily_vol = ml.util.get_daily_vol(close=datasets['Adj Close'], lookback=40)

# Apply Symmetric CUSUM Filter and get timestamps for events
# Note: Only the CUSUM filter needs a point estimate for volatility
cusum_events = ml.filters.cusum_filter(datasets['Adj Close'], threshold=daily_vol.mean() * 0.5)

t_events = cusum_events

# Compute vertical barrier
vertical_barriers = ml.labeling.add_vertical_barrier(t_events=t_events, close=datasets['Adj Close'], num_days=1)

pt_sl = [1, 2]
min_ret = 0.005
triple_barrier_events = ml.labeling.get_events(close=datasets['Adj Close'],
                                               t_events=t_events,
                                               pt_sl=pt_sl,
                                               target=daily_vol,
                                               min_ret=min_ret,
                                               num_threads=3,
                                               vertical_barrier_times=vertical_barriers,
                                               side_prediction=datasets['side'])
labels = ml.labeling.get_bins(triple_barrier_events, datasets['Adj Close'])
labels.side.value_counts()


print2(labels.head())

daily_vol.plot()
plt.axhline(daily_vol.mean()*2.0, color="g")
plt.axhline(daily_vol.mean(), color="y")
plt.axhline(daily_vol.mean()*0.5, color="r");
