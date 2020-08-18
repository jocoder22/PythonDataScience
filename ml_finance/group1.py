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

import scipy.stats

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
stddevs = datasets['Adj Close'].std()
print2(means, stddevs)

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
normalised_returns = (datasets['Adj Close'].pct_change() - means)/stddevs

matplotlib.rcParams['figure.figsize'] = [12, 7]
plt.plot(normalised_prices);
plt.ylabel('Normalised Price');
plt.legend(assets);
plt.grid()
show

"""


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
print2(model_fit.summary())


residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print2(residuals.describe())


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
print2('Test MSE: %.3f' % error)


plt.plot(test)
plt.plot(predictions, color='red')
show()


# rolling window volatililty
mr =  datasets['Adj Close']["MSFT"].pct_change().dropna()


rolling = mr.rolling(30)
vol = rolling.std().dropna()
vol_monthly = vol.resample("M").mean()
vol_monthly.plot(title="Monthly volatility").set_ylabel("Standard Deviation")
show();


vol_monthly.pct_change().plot(title="$\Delta$ Monthly volatility").set_ylabel("$\Delta$ Standard Deviation")
show();



qmin = mr.resample("Q").min().dropna()
vol_q = vol.resample("Q").mean().dropna()

# Create a plot of quarterly minimum portfolio returns
plt.plot(qmin, label="Quarterly minimum return")

# Create a plot of quarterly mean volatility
plt.plot(vol_q, label="Quarterly mean volatility")



# Create legend and plot
# Create legend and plot
plt.legend()
plt.show()

"""


import statsmodels.api as sm

"""
# Add intercept constants to each sub-period 'before' and 'after'
before = mr.loc[:"2007"]
after = mr.loc["2008":]

kmr = np.ones([mr.shape[0]])
kkb = np.ones([before.shape[0]])
kka = np.ones([after.shape[0]])

mr_intercept  = sm.add_constant(kmr)
before_with_intercept = sm.add_constant(kkb)
after_with_intercept  = sm.add_constant(kka)


# Fit OLS regressions to each tota; period
result = sm.OLS(mr, mr_intercept).fit()

# # Retrieve the sum-of-squared residuals
ssr_total = result.ssr


# Fit OLS regressions to each sub-period
r_b = sm.OLS(before, before_with_intercept).fit()
r_a = sm.OLS(after,  after_with_intercept).fit()

# # Retrieve the sum-of-squared residuals
ssr_before = r_b.ssr
ssr_after = r_a.ssr

# # Fit OLS regressions to each total period
# result = sm.RLM(mr, mr_intercept, M=sm.robust.norms.HuberT()).fit()

# # Retrieve the sum-of-squared residuals
# ssr_total = np.power(result.resid, 2).sum()

# # Fit OLS regressions to each sub-period
# r_b = sm.RLM(before, before_with_intercept, M=sm.robust.norms.HuberT()).fit()
# r_a = sm.RLM(after,  after_with_intercept, M=sm.robust.norms.HuberT()).fit()

# # Get sum-of-squared residuals for both regressions
# ssr_before = np.power(r_b.resid, 2).sum()
# ssr_after = np.power(r_a.resid, 2).sum()

# Compute and display the Chow test statistic
d_f = 1
df2 = 2*d_f
numerator = ((ssr_total - (ssr_before + ssr_after)) / d_f)
denominator = ((ssr_before + ssr_after) / (mr.shape[0] - df2))
print("Chow test statistic: ", numerator / denominator)

scipy.stats.f.ppf(q=1-0.01, dfn=d_f, dfd=(mr.shape[0] - df2))

"""
# print2(normalised_prices.head())
# mr = normalised_prices["MSFT"]

# print2(normalised_returns.head())
# mr = normalised_returns["MSFT"].dropna()


mmr = normalised_prices.resample("Q").mean()
print2(mmr.head())
mr = mmr["MSFT"].dropna()

print2(mr.head(), mr.shape)
before = mr.loc[:"2016-08"]
after = mr.loc["2016-09":]
print2("#"*20, before.tail())


kmr = np.ones([mr.shape[0]])
kkb = np.ones([before.shape[0]])
kka = np.ones([after.shape[0]])

mr_intercept  = sm.add_constant(kmr)
before_with_intercept = sm.add_constant(kkb)
after_with_intercept  = sm.add_constant(kka)



# Fit OLS regressions to each tota; period
result = sm.OLS(mr, mr_intercept).fit()

# # Retrieve the sum-of-squared residuals
ssr_total = result.ssr


# Fit OLS regressions to each sub-period
r_b = sm.OLS(before, before_with_intercept).fit()
r_a = sm.OLS(after,  after_with_intercept).fit()

# # Retrieve the sum-of-squared residuals
ssr_before = r_b.ssr
ssr_after = r_a.ssr



"""
# Fit OLS regressions to each total period
result = sm.RLM(mr, mr_intercept, M=sm.robust.norms.HuberT()).fit()

# Retrieve the sum-of-squared residuals
ssr_total = np.power(result.resid, 2).sum()

# Fit OLS regressions to each sub-period
r_b = sm.RLM(before, before_with_intercept, M=sm.robust.norms.HuberT()).fit()
r_a = sm.RLM(after,  after_with_intercept, M=sm.robust.norms.HuberT()).fit()

# Get sum-of-squared residuals for both regressions
ssr_before = np.power(r_b.resid, 2).sum()
ssr_after = np.power(r_a.resid, 2).sum()

"""




# Compute and display the Chow test statistic
d_f = 1
df2 = 2*d_f
numerator = ((ssr_total - (ssr_before + ssr_after)) / d_f)
denominator = ((ssr_before + ssr_after) / (mr.shape[0] - df2))
print("Chow test statistic: ", numerator / denominator)

f = scipy.stats.f.ppf(q=1-0.01, dfn=d_f, dfd=(mr.shape[0] - df2))
print2(f"F Critical point: {f}")
