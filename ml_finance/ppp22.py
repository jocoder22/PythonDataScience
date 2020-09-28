import pandas as pd
import pandas_datareader.data as dr
import matplotlib
import matplotlib.pyplot as plt
import math
import statsmodels.tsa.stattools as ts
import numpy as np
from scipy import stats

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import mlfinlab as ml
# from mlfinlab.feature_importance import ClassificationModelFingerprint
# from mlfinlab.feature_importance import plot_feature_importance
# from mlfinlab.feature_importance import (feature_importance_mean_decrease_impurity, feature_importance_mean_decrease_accuracy, feature_importance_sfi, plot_feature_importance)
# from mlfinlab.feature_importance import ClassificationModelFingerprint


from printdescribe import  print2
import warnings
warnings.filterwarnings('ignore')


start_date = '2013-01-01'
# end_date = '2020-02-29'

# Download daily Amazon stock Adjusted close prices and indexes
assets = ['AMZN', "^GSPC", "^DJI", "^IXIC", "^RUT", "CL=F"]
datasets = dr.DataReader(assets, data_source='yahoo', start = start_date)["Adj Close"]
datasets.tail()

# Name of the columns
col = ["Amazon", "Sp500", "Dow20", "Nasdaq", "R2000", "Crude20"]
datasets.columns = col
print2(datasets.head())

datasets.iloc[:, ~datasets.columns.isin(["Dow20", "Nasdaq"])].plot(figsize=(10,5));
data = datasets.copy()
data['close'] = data["Amazon"]

# compute moving averages
fast_window = 20
slow_window = 50

data['fast_mavg'] = data['close'].rolling(window=fast_window, min_periods=fast_window, center=False).mean()
data['slow_mavg'] = data['close'].rolling(window=slow_window, min_periods=slow_window, center=False).mean()
data.head()

# Compute sides
data['side'] = np.nan

long_signals = data['fast_mavg'] >= data['slow_mavg'] 
short_signals = data['fast_mavg'] < data['slow_mavg'] 
data.loc[long_signals, 'side'] = 1
data.loc[short_signals, 'side'] = -1

# Remove Look ahead bias by lagging the signal
data['side'] = data['side'].shift(1)

daily_vol = ml.util.get_daily_vol(close=data['close'], lookback=50)
cusum_events = ml.filters.cusum_filter(data['close'], threshold=daily_vol.mean() * 0.5)

t_events = cusum_events
vertical_barriers = ml.labeling.add_vertical_barrier(t_events=t_events, close=data['close'], num_days=1)

pt_sl = [1, 2]
min_ret = 0.005
triple_barrier_events = ml.labeling.get_events(close=data['close'],
                                               t_events=t_events,
                                               pt_sl=pt_sl,
                                               target=daily_vol,
                                               min_ret=min_ret,
                                               num_threads=3,
                                               vertical_barrier_times=vertical_barriers,
                                               side_prediction=data['side'])
labels = ml.labeling.get_bins(triple_barrier_events, data['close'])


# X = pd.DataFrame(index=data.index)

X = data.iloc[:,1:]
X.head()


# Volatility
data['log_ret'] = np.log(data['close']).diff()
X['volatility_50'] = data['log_ret'].rolling(window=50, min_periods=50, center=False).std()
X['volatility_31'] = data['log_ret'].rolling(window=31, min_periods=31, center=False).std()
X['volatility_15'] = data['log_ret'].rolling(window=15, min_periods=15, center=False).std()

# Autocorrelation
window_autocorr = 50

X['autocorr_1'] = data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=1), raw=False)
X['autocorr_2'] = data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=2), raw=False)
X['autocorr_3'] = data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=3), raw=False)
X['autocorr_4'] = data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=4), raw=False)
X['autocorr_5'] = data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=5), raw=False)


# Log-return momentum
X['log_t1'] = data['log_ret'].shift(1)
X['log_t2'] = data['log_ret'].shift(2)
X['log_t3'] = data['log_ret'].shift(3)
X['log_t4'] = data['log_ret'].shift(4)
X['log_t5'] = data['log_ret'].shift(5)

X.dropna(inplace=True)

labels = labels.loc[X.index.min():X.index.max(), ]
triple_barrier_events = triple_barrier_events.loc[X.index.min():X.index.max(), ]


X = X.loc[X.index.isin(labels.index),:]
X.drop("close", axis=1, inplace=True)
X_train, _ = X.iloc[:], X.iloc[:] # take all values for this example
y_train = labels.loc[X_train.index, 'bin']

X_train.shape, y_train.shape

X_train.head()


# instantiate Decision tree model
base_estimator = DecisionTreeClassifier(class_weight = 'balanced', random_state=42,
                                        max_depth=10, criterion='entropy',
                                        min_samples_leaf=4, min_samples_split=3, max_features='auto')
clf = BaggingClassifier(n_estimators=452, n_jobs=-1, random_state=42, oob_score=True, base_estimator=base_estimator)
clf.fit(X_train, y_train)


cv_gen =  ml.cross_validation.PurgedKFold(n_splits=5, samples_info_sets=triple_barrier_events.loc[X_train.index].t1, pct_embargo = 0.02)
cv_score_acc = ml.cross_validation.ml_cross_val_score(clf, X_train, y_train, cv_gen, scoring=accuracy_score)
cv_score_f1 = ml.cross_validation.ml_cross_val_score(clf, X_train, y_train, cv_gen, scoring=f1_score)

cv_gen =  ml.cross_validation.PurgedKFold(n_splits=5, samples_info_sets=triple_barrier_events.loc[X_train.index].t1, pct_embargo = 0.02)
cv_score_acc = ml.cross_validation.ml_cross_val_score(clf, X_train, y_train, cv_gen, scoring=accuracy_score)
cv_score_f1 = ml.cross_validation.ml_cross_val_score(clf, X_train, y_train, cv_gen, scoring=f1_score)

labels = labels.loc[X.index.min():X.index.max(), ]
triple_barrier_events = triple_barrier_events.loc[X.index.min():X.index.max(), ]

X = X.loc[X.index.isin(labels.index),:]

# X.drop("close", axis=1, inplace=True)

X_train, _ = X.iloc[:], X.iloc[:] # take all values for this example
y_train = labels.loc[X_train.index, 'bin']

X_train.shape, y_train.shape

base_estimator = DecisionTreeClassifier(class_weight = 'balanced', random_state=42,
                                        max_depth=10, criterion='entropy',
                                        min_samples_leaf=4, min_samples_split=3, max_features='auto')
clf = BaggingClassifier(n_estimators=452, n_jobs=-1, random_state=42, oob_score=True, base_estimator=base_estimator)
clf.fit(X_train, y_train)

cv_gen =  ml.cross_validation.PurgedKFold(n_splits=5, samples_info_sets=triple_barrier_events.loc[X_train.index].t1, pct_embargo = 0.02)
cv_score_acc = ml.cross_validation.ml_cross_val_score(clf, X_train, y_train, cv_gen, scoring=accuracy_score)
cv_score_f1 = ml.cross_validation.ml_cross_val_score(clf, X_train, y_train, cv_gen, scoring=f1_score)



clf_fingerpint = ClassificationModelFingerprint()
clf.fit(X_train, y_train)

feature_combinations = [('volatility_50', 'volatility_15'), ('volatility_50', 'autocorr_4'), 
                       ('autocorr_1', 'autocorr_4'), ('autocorr_4', 'volatility_15'), ('volatility_50', 'log_t4'),
                       ('volatility_31', ('autocorr_4'))]

clf_fingerpint.fit(clf, X_train, num_values=50, pairwise_combinations=feature_combinations)


# Plot linear effect
plt.figure(figsize=(17, 12))
plt.title('Model linear effect')
plt.bar(*zip(*clf_fingerpint.linear_effect['raw'].items()))
# plt.savefig('linear.png')

# Plot non-linear effect
plt.figure(figsize=(17, 12))
plt.title('Model non-linear effect')
plt.bar(*zip(*clf_fingerpint.non_linear_effect['raw'].items()))
# plt.savefig('non_linear.png')

# Plot pairwise effect
plt.figure(figsize=(17, 12))
plt.title('Model pairwise effect')
plt.bar(*zip(*clf_fingerpint.pair_wise_effect['raw'].items()))
plt.show()
# plt.savefig('pairwise.png')
