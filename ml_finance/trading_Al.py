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
import pyfolio as pf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve




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
plt.axhline(daily_vol.mean()*0.5, color="r")
plt.show();


datasets['dv'] = daily_vol
datasets['dv'].fillna(0, inplace=True)
datasets['upper'] = datasets['Adj Close'] + datasets['dv'] * 1000
datasets['lower'] = datasets['Adj Close'] - datasets['dv'] * 1000

datasets['Adj Close'].plot(figsize=(10,10))
datasets['upper'].plot(c="g")
datasets['lower'].plot(c="r")
# (datasets['side2']).dropna().plot(c='black');


primary_forecast = pd.DataFrame(labels['bin'])
primary_forecast['pred'] = 1
primary_forecast.columns = ['actual', 'pred']

# Performance Metrics
actual = primary_forecast['actual']
pred = primary_forecast['pred']
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print('')
print("Accuracy")
print(accuracy_score(actual, pred))


# Momentum: Price Rate of Change, Moving Average Convergence-Divergence, Macd Threshould
# Volatility: Rolling Standard Deviation
# Correlation: Serial (Auto) Correlation
# Returns: Log_returns

# Feature generation

datasets = orig_data

datasets['log_ret'] = np.log(datasets['Adj Close']).diff()

# Momentum
datasets['mom2'] = datasets['Adj Close'].pct_change(periods=2)

## Moving Average Convergence-Divergence ##

ewma_26 = datasets['Adj Close'].transform(lambda x: x.ewm(span = 26).mean())
ewma_12 = datasets['Adj Close'].transform(lambda x: x.ewm(span = 12).mean())

macd = ewma_12 - ewma_26

threshold = macd.ewm(span = 9).mean()

datasets['cd threshold'] = threshold
datasets['MACD'] = macd

# Volatility
datasets['volatility_5'] = datasets['log_ret'].rolling(window=5, min_periods=5, center=False).std()

# Serial Correlation (Takes about 4 minutes)
window_autocorr = 10

datasets['autocorr_2'] = datasets['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=2), raw=False)

# Get the various log -t returns
datasets['log_t1'] = datasets['log_ret'].shift(1)

# Re compute sides
datasets['side'] = np.nan

long_signals = datasets['fast_mavg'] >= datasets['slow_mavg']
short_signals = datasets['fast_mavg'] < datasets['slow_mavg']

datasets.loc[long_signals, 'side'] = 1
datasets.loc[short_signals, 'side'] = -1

# Remove look ahead bias
datasets = datasets.shift(1)

# Get features at event dates
X = datasets.loc[labels.index, :]

# Drop unwanted columns
X.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'fast_mavg', 'slow_mavg', 'side'], axis=1, inplace=True)

y = labels['bin']


# Split data into training, validation and test sets
X_training_validation = X[start_date:end_date]
y_training_validation = y[start_date:end_date]
X_train, X_validate, y_train, y_validate = train_test_split(X_training_validation, y_training_validation, test_size=0.30, shuffle=False)

train_df = pd.concat([y_train, X_train], axis=1, join='inner')

train_df['bin'].value_counts()


# Upsample the training data to have a 50 - 50 split
# https://elitedatascience.com/imbalanced-classes
majority = train_df[train_df['bin'] == 0]
minority = train_df[train_df['bin'] == 1]

new_minority = resample(minority, 
                   replace=True,     # sample with replacement
                   n_samples=majority.shape[0],    # to match majority class
                   random_state=42)

train_df = pd.concat([majority, new_minority])
train_df = shuffle(train_df, random_state=42)

train_df['bin'].value_counts()


# Create training data
y_train = train_df['bin']
X_train= train_df.loc[:, train_df.columns != 'bin']




nest = [int(x) for x in np.linspace(100,600,6)]

parameters = {'max_depth':list(range(10,17)),
              'n_estimators':nest,
              'random_state':[42]}
    
def perform_grid_search(X_data, y_data):
    rf = RandomForestClassifier(criterion='entropy')
    
    clf = GridSearchCV(rf, parameters, cv=4, scoring='roc_auc', n_jobs=-1)
    
    clf.fit(X_data, y_data)
    
    print(clf.cv_results_['mean_test_score'])
    
    return clf.best_params_['n_estimators'], clf.best_params_['max_depth']
  
  # extract parameters
# n_estimator, depth = perform_grid_search(X_train, y_train)
c_random_state = 42
# print(n_estimator, depth, c_random_state)
# n_estimator, depth = 100, 15
n_estimator, depth = 100, 15



# Performance Metrics
y_pred_rf = rf.predict_proba(X_train)[:, 1]
y_pred = rf.predict(X_train)
fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)
print(classification_report(y_train, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_train, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_train, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# Meta-label
# Performance Metrics
y_pred_rf = rf.predict_proba(X_validate)[:, 1]
y_pred = rf.predict(X_validate)
fpr_rf, tpr_rf, _ = roc_curve(y_validate, y_pred_rf)
print(classification_report(y_validate, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_validate, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_validate, y_pred))


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF ROC')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


x_validate_min = X_validate.index.min()
x_validate_max = X_validate.index.max()
print(x_validate_min)
print(x_validate_max)


# Primary model
primary_forecast = pd.DataFrame(labels['bin'])
primary_forecast['pred'] = 1
primary_forecast.columns = ['actual', 'pred']

start = primary_forecast.index.get_loc(x_validate_min)
end = primary_forecast.index.get_loc(x_validate_max) + 1

subset_prim = primary_forecast[start:end]

# Performance Metrics
actual = subset_prim['actual']
pred = subset_prim['pred']
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print('')
print("Accuracy")
print(accuracy_score(actual, pred))

# Feature Importance
title = 'Feature Importance:'
figsize = (15, 5)

feat_imp = pd.DataFrame({'Importance':rf.feature_importances_})    
feat_imp['feature'] = X.columns
feat_imp.sort_values(by='Importance', ascending=False, inplace=True)
feat_imp = feat_imp

feat_imp.sort_values(by='Importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title=title, figsize=figsize)
plt.xlabel('Feature Importance Score')
plt.show()



def get_daily_returns(intraday_returns):
    """
    This changes returns into daily returns that will work using pyfolio.
    """
    day_val_data = 3
    intra_day_returns = intraday_returns/day_val_data
    
    cum_rets = ((intra_day_returns + 1).cumprod())

    # Downsample to daily
    daily_rets = cum_rets.resample('B').last()
    
    

    # Forward fill, Percent Change, Drop NaN
    daily_rets = daily_rets.ffill().pct_change().dropna()
    
    return daily_rets

  
valid_dates = X_validate.index
base_rets = labels.loc[valid_dates, 'ret']
primary_model_rets = get_daily_returns(base_rets)

# Set-up the function to extract the KPIs from pyfolio
perf_func = pf.timeseries.perf_stats

# Save the statistics in a dataframe
perf_stats_all = perf_func(returns=primary_model_rets, 
                           factor_returns=None, 
                           positions=None,
                           transactions=None,
                           turnover_denom="AGB")
perf_stats_df = pd.DataFrame(data=perf_stats_all, columns=['Primary Model'])

pf.show_perf_stats(primary_model_rets)


meta_returns = labels.loc[valid_dates, 'ret'] * y_pred
daily_meta_rets = get_daily_returns(meta_returns)

# Save the KPIs in a dataframe
perf_stats_all = perf_func(returns=daily_meta_rets, 
                           factor_returns=None, 
                           positions=None,
                           transactions=None,
                           turnover_denom="AGB")

perf_stats_df['Meta Model'] = perf_stats_all

pf.show_perf_stats(daily_meta_rets)


# Define a trailing 252 trading day window
window = 252

# Calculate the max drawdown in the past window days for each day 
rolling_max = datasets['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = datasets['Adj Close']/rolling_max - 1.0

# Calculate the minimum (negative) daily drawdown
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

# Plot the results
daily_drawdown.plot()
max_daily_drawdown.plot()

# Show the plot
plt.show()


# pf.create_returns_tear_sheet(daily_meta_rets)
