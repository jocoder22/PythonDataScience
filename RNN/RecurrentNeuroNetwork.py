#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import datetime
# import tensorflow as tf



from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import pandas_datareader as pdr
import seaborn as sns

# plt.style.use('ggplot')

def train_validate_test_split(dataset, train_percent=.6, validate_percent=.2):
    np.random.seed(3456)
    perm = np.random.permutation(dataset.index)
    m = len(perm)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = dataset.ix[perm[:train_end]]
    validate = dataset.ix[perm[train_end:validate_end]]
    test = dataset.ix[perm[validate_end:]]
    
    return train, validate, test


def train_validate_test_split2(datatt, tx, vx):
    vxx = tx + vx
    train, validate, test = np.split(
        datatt.sample(frac=1), [int(.tx*len(datatt)), int(.vxx*len(datatt))])

    return train, validation, test


symbol = 'RELIANCE.NS'
starttime = datetime.datetime(1996, 1, 1)
endtime = datetime.datetime(2018, 9, 30)
rel = pdr.get_data_yahoo(symbol, starttime, endtime)[['Open','High', 'Low', 'Close']]
print(rel.head())


# https://www.youtube.com/watch?v=dNFgRUD2w68

# Visualizations
pd.plotting.scatter_matrix(rel, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
plt.show()
sns.pairplot(rel)
plt.show()

corr = rel.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=False)
plt.show()

# Scale data
minmaxscaler = MinMaxScaler()
standscaler = StandardScaler()

# Because the distribution does not approx normal, the MinMaxScaler will be better
mmdata = minmaxscaler.fit_transform(rel)
staddata = standscaler.fit_transform(rel)

X = rel.drop('Close', axis=1)
y = rel[['Close']]
# Create training, validataion and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
