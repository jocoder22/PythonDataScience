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


# Obtaining the mean and standard devivation of the Assests
means = datasets['Adj Close'].mean()
print(means)
print("............")
stddevs = datasets['Adj Close'].std()
print(stddevs)

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
  show


  # Normalizing prices
normalised_prices = (datasets['Adj Close'] - means)/stddevs

matplotlib.rcParams['figure.figsize'] = [12, 7]
plt.plot(normalised_prices);
plt.ylabel('Normalised Price');
plt.legend(assets);
plt.grid()
show

# Identifying and Testing for Structural Breaks
def no_structural_breaks_test(ticker, benchmark, data):
  results = smf.ols(ticker + ' ~ ' + benchmark, data = data).fit()
  names = ['test statistic', 'pval', 'crit']
  test = breaks_cusumolsresid(results.resid, results.df_model)
  print('Test for the null-hypothesis of no structural breaks for '+ ticker + ' vs ' + benchmark + ':')
  print(list(zip(names, test)))
  print()

no_structural_breaks_test('MSFT', 'FDN', normalised_prices)
no_structural_breaks_test('MSFT', 'JPM', normalised_prices)
no_structural_breaks_test('MSFT', 'XLF', normalised_prices)

# Applying the Jarque Bera Test
log_returns = np.log(datasets['Adj Close']).diff().dropna()

for ticker in assets:
  print('Jarque-Bera test for ' + ticker + ': ', jarque_bera(log_returns[ticker]))


# Applying a Cointegration Test
def cointegration_test(ticker, benchmark, data):
  johansen_results = coint_johansen(normalised_prices[[ticker, benchmark]], 0, 1)
  print(ticker, ': ') 
  print('Maximum eigenvalue statistic:', johansen_results.lr2)
  print('Critical values (90%, 95%, 99%) for maximum eigenvalue statistic', johansen_results.cvm)
  print()

cointegration_test('MSFT', 'FDN', normalised_prices)
cointegration_test('MSFT', 'JPM', normalised_prices)
cointegration_test('MSFT', 'XLF', normalised_prices)


# Applying an AR(1) Model and Forecasting for the next period (t+1)
model = ARIMA(log_returns['MSFT'], order=(1,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

X = log_returns['MSFT'].values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)+1]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(1,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()