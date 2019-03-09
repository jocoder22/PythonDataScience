#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
plt.style.use('ggplot')


path = r'C:\Users\Jose\Desktop\PythonDataScience\tensorflow\Array'
os.chdir(path)

sp = '\n\n'
data = pd.read_csv('train.csv')

print(data.columns.tolist())

print(data.isnull().values.any(), end=sp)
# print(data.isnull().sum().tolist(), end=sp)

# show the rows index with missing values
# print(data.isnull().index.tolist())
# print(data.loc[data.isnull().sum(1)>1].index.tolist())
# print(data.loc[data[['V4', 'V5', 'V6', 'V7', 'V8','V14', 'V15', 'V16','V17', 'V18', 'V19']].isnull().any(1)].index)

# print the columns with missing values
# print(data.columns[data.isnull().any()].tolist())
# print(data.head(), data.shape, data.info(), sep=sp)
# print(data[['V8','V14', 'V15', 'V16','V17', 'V18', 'V19']].tail(), end=sp)

data.dropna(subset=['V7', 'V17'], inplace=True)
print(data.isnull().values.any(), end=sp)


# Get the features and labels
xd = data.drop(columns=['Class'])
ycat = data[['Class']]


scaler = MinMaxScaler()
onehot = OneHotEncoder(sparse=False, categories='auto')

xdata = scaler.fit_transform(xd)
x = np.array(xdata)
y = onehot.fit_transform(ycat)
print(y.shape, type(y), sep=sp)

print(y[:5])

ydata = pd.DataFrame(y, columns=['FF', 'NF'])
print(ydata.head())