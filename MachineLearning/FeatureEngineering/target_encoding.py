#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from contextlib import contextmanager

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

sp = {'sep':'\n\n', 'end':'\n\n'}
path = r'C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\FeatureEngineering'

os.chdir(path)
df = pd.read_csv('housing.csv')

kfold = KFold(n_splits=4, shuffle=True, random_state=1973)



def test_encoding(train, test, target, cat, alpha=7):
    # global mean on the train data
    mean_global = train[target].mean()
    
    # Get categorical feature sum and size
    cat_sum = train.groupby(cat)[target].sum()
    cat_size = train.groupby(cat).size()
    
    # smoothed  statistics
    train_smoothed = (cat_sum + mean_global * alpha) / (cat_size + alpha)
    
    # get encodings for  test data
    test_encoded = test[cat].map(train_smoothed).fillna(mean_global)
    return test_encoded.values



def train_encoding(train, target, cat, alpha=7):
    # 4-fold cross-validation
    k_fold = KFold(n_splits=4, random_state=1973, shuffle=True)
    feature_t = pd.Series(index=train.index)
    
    # train k-fold encoding
    for train_index, test_index in k_fold.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
      
        # out-of-fold statistics and apply to cv_test
        cv_test_feature = test_encoding(cv_train, cv_test, target, cat, alpha)
        
        # create new train feature for the fold
        feature_t.iloc[test_index] = cv_test_feature
        
    return feature_t.values


def target_encoding(train, test, target, cat, alpha=7):
    
    # test data mean target coded feature
    test_mean_coded = test_encoding(train, test, target, cat, alpha)

    # train data mean target coded feature
    train_mean_coded = train_encoding(train, target, cat, alpha)

    
    # Return new features to add to the model
    return train_mean_coded, test_mean_coded





