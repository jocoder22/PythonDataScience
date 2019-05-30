#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.pipeline import FeatureUnion
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.preprocessing import Imputer
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# plt.style.use('ggplot')


sp = '\n\n'
url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00488/Live.csv'

df = pd.read_csv(url2)
df_int = df.select_dtypes(include='object')

print(df_int['status_type'].value_counts())


# One Hot coding
df_hot = pd.get_dummies(df, columns=['status_type'], prefix='S')
print(df_hot.head())

# Dummy coding
df_dummy = pd.get_dummies(df, columns=['status_type'], prefix='D', drop_first=True)
print(df_dummy.head())


print(df.columns, df_dummy.columns, sep=sp)


# Collapsing values