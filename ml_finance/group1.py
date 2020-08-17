# Importing Libraries 
import pandas as pd 
import pandas_datareader.data as dr
import matplotlib
import matplotlib.pyplot as plt 
import math
import statsmodels.tsa.stattools as ts
import numpy as np 
from scipy import stats
from scipy.stats import jarque_bera
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from printdescribe import print2
show = plt.show

# Obtaining Stock Data of Microsoft and Benchmark Data
start_date = '2013-01-01'
end_date = '2019-12-31'

assets = ['MSFT', 'FDN', 'JPM', 'XLF']

datasets = dr.DataReader(assets, data_source='yahoo', start=start_date, end=end_date)

print2(datasets['Adj Close'].head())


# matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(datasets['Adj Close'])
plt.ylabel('Price')
plt.legend(assets)
plt.grid()
show()

# Obtaining the 30-day moving average and exponentially weighted moving average
moving_average = datasets['Adj Close'].rolling(window = 30).mean()
ewma = datasets['Adj Close'].ewm(span=30).mean()

figure_counter = 0
matplotlib.rcParams['figure.figsize'] = [12, 5]
for ticker in assets:
  figure_counter = figure_counter + 1
  plt.figure(figure_counter)
  plt.plot(datasets['Adj Close'][ticker])
  plt.plot(moving_average[ticker], '--')
  plt.plot(ewma[ticker], '--')
  plt.legend(['Price', '30-day mov avg','30-day EWM avg'])
  plt.title(ticker)
  plt.grid()
  show()


  # Normalizing prices
normalised_prices = (datasets['Adj Close'] - means)/stddevs

matplotlib.rcParams['figure.figsize'] = [12, 7]
plt.plot(normalised_prices);
plt.ylabel('Normalised Price');
plt.legend(assets);
plt.grid()
show()

