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