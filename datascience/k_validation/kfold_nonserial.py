#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from contextlib import contextmanager

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from printdescribe import print2, describe2, changepath

@contextmanager
def changepath(path):
    currentpath = os.getcwd()

    os.chdir(path)

    try:
        yield 

    finally:
        os.chdir(currentpath)


# model = DecisionTreeRegressor(min_samples_leaf = 11 , min_samples_split = 33, random_state=500)

plt.style.use('ggplot')
path = 'C:\\Users\\Jose\\Desktop\\TimerSeriesAnalysis'
sp = {'sep':'\n\n', 'end':'\n\n'}


with changepath(path):
    df = pd.read_csv('AMZN.csv')


print(df.head(), df.info(), **sp)

# scale the dataset
scaler = StandardScaler()

# # First fit and transform dataset on training .
# df_X = scaler.fit_transform(df.loc[:,['Open', 'Volume']])
# df22 = pd.DataFrame(df_X, columns=['Open', 'Volume'])
# y = df.loc[:,['Close']]

# df2 = pd.concat([df22, y], axis=1, sort=False )


kf_object = KFold(n_splits=5, shuffle=False, random_state=1973)

k_fold = 0
model_scores = []
model_rmse = []
for train_idx, test_idx in kf_object.split(df):
    train_cv, test_cv = df.iloc[train_idx], df.iloc[test_idx]
    print(f'Fold: {k_fold}')
    print(f'Train fold shape: {train_cv.shape}')
    print(f'Train fold range: {train_cv.index.min()} - {train_cv.index.max()}')
    print(f'Test fold range: {test_cv.index.min()} - {test_cv.index.max()}', **sp)
  
    k_fold += 1
    model = DecisionTreeRegressor(random_state=500)

    model.fit(train_cv[['Open', 'Volume']], train_cv['Close'])

    pred = model.predict(test_cv[['Open', 'Volume']])
    score = model.score(test_cv[['Open', 'Volume']], test_cv['Close'])
    rmse = np.sqrt(mean_squared_error(test_cv['Close'], pred))
    model_scores.append(score)
    model_rmse.append(rmse)


mean_score = np.mean(model_scores) * 100
mean_rmse = np.mean(model_rmse)

overall_score = mean_score - np.std(model_scores) 
overall_rmse = mean_rmse + np.std(model_rmse)

print(f"{'Mean':>18} {'OverallScore':^21}")
print(f'Accuracy:    {mean_score:.02f}     {overall_score:.02f}')
print(f'RMSE    :    {mean_rmse:.02f}      {overall_rmse:.02f}')
print(model_scores, model_rmse, **sp)



def get_fold_mse(train, kf):
    mse_scores = []
    mse_r2 = []
    
    for train_index, test_index in kf.split(train):
        fold_train, fold_test = train.loc[train_index], train.loc[test_index]

        # Fit the data and make predictions
        # Create a Random Forest object
        # rf = RandomForestRegressor(n_estimators=10, random_state=123)
        rf = LinearRegression(normalize=True)
        # Train a model
        rf.fit(X=fold_train[["Open", "Volume"]], y=fold_train['Close'])

        # Get predictions for the test set
        pred = rf.predict(fold_test[["Open", "Volume"]])

        rr = rf.score(fold_test[["Open", "Volume"]], fold_test['Close'])

        fold_score = round(np.sqrt(mean_squared_error(fold_test['Close'], pred)), 5)
        mse_scores.append(fold_score)
        mse_r2.append(rr)
        
    return mse_scores, mse_r2



mse_scores, r2_scores = get_fold_mse(df, kf_object)

print('Mean validation RMSE: {:.5f}'.format(np.mean(mse_scores)))
print('Mean validation R^2: {:.5f}'.format(np.mean(r2_scores)))

