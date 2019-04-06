#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import pickle as pkl

from joblib import dump
from datetime import datetime, date

from sklearn.ensemble import RandomForestRegressor as regg
from sklearn.model_selection import train_test_split as tts

plt.style.use('ggplot')

path = r'C:\Users\Jose\Desktop\PythonDataScience\financial_AI'
os.chdir(path)

sp = '\n\n'
dat1 = date.today()
datenow = dat1.strftime('%b_%d_%y')

start = datetime(2000, 6, 29)
# end = datetime(2018, 3, 27)
end = date.today()
# symbol = 'AAPL'
# symbol = 'AMZN'
symbol = 'NYT'

stock = pdr.get_data_yahoo(symbol, start, end)[['Close']]

for d in range(1, 41):
    dd = 'day' + str(d)
    stock[dd] = stock['Close'].shift(-1 * d)

stock.dropna(inplace=True)

print(stock.head(), stock.tail(), sep=sp, end=sp)

X = stock.iloc[:, :33]
y = stock.iloc[:, 33:]

print(X.shape, y.shape, sep=sp, end=sp)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, shuffle=False)

# loading the model
# symbol = 'AAPL'
# filename = f'model_{symbol}_{datenow}.sav'
# model = pkl.load(open(filename, 'rb'))
# result = model.score(X_test, y_test)
# print(result)

model = regg(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=130, n_jobs=16,
           oob_score=True, random_state=None, verbose=1, warm_start=False)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score, sep=sp, end=sp)

impt_feature = model.feature_importances_
print(impt_feature)

ypred = model.predict(X_test)

fig, ax = plt.subplots()

fig.set_figwidth(24)
fig.set_figheight(15)

ax.plot(y_test.index, y_test['day40'], 'ro', label='Actual')
ax.plot(y_test.index, ypred[:, 7], 'bo', label='Prediction')
plt.legend()
plt.show()

y_test2 = y_test.copy()
y_test2['pred'] = ypred[:, 7]
fig, ax = plt.subplots()

fig.set_figwidth(24)
fig.set_figheight(15)

y_test['day40'].plot(label='Actual')
y_test2['pred'].plot(label='Prediction')
plt.legend()
plt.show()

y_test2.sort_index(inplace=True)
print(y_test2.tail(10))

# # Save the model
filename = f'model_{symbol}_{datenow}.sav'
pkl.dump(model, open(filename, 'wb'))
dump(model, f'model2_{symbol}_{datenow}.joblib')


# # loading the model
# model2 = pkl.load(open(filename, 'rb'))
# result = model2.score(X_test, y_test)
# print(result)

