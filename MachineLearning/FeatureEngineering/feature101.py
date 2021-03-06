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


sp = {'sep':'\n\n', 'end':'\n\n'}
url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00488/Live.csv'

df = pd.read_csv(url2)
df_int = df.select_dtypes(include='object')
print(df.head(), df.columns, **sp)
print(df_int['status_type'].value_counts(),  **sp)


# One Hot coding
df_hot = pd.get_dummies(df, columns=['status_type'], prefix='S')
print(df_hot.head(), df_hot.columns, **sp)
print(df.head(), **sp)



# Dummy coding
df_dummy = pd.get_dummies(df, columns=['status_type'], prefix='D', drop_first=True)
print(df_dummy.head())
print(df.columns, df_dummy.columns, **sp)


# Collapsing values
df['status_type2'] = df['status_type']
mask = df['status_type'].isin(['status', 'link'])
df['status_type'][mask] = 'Others'
counts = df['status_type'].value_counts()
print(counts, end=sp)
# print(df['num_shares'].value_counts(), **sp)

# pandas.where => Replace values where the condition is False.
df['status_type2'] = df['status_type2'].where(lambda x: ~x.isin(['status', 'link']), 'Others')
print(df['status_type2'].value_counts(), **sp)

# creating categories
# Binning feature
# cut_points = [-np.inf, 100, 500, 1000,  np.inf]
minn, maxx = (df["num_shares"].min(), df["num_shares"].max())
cut_points = [minn, 100, 500, 1000, maxx]
labels = ["low", "medium", "high", "very high"]
df["num_shares_Bin"] = pd.cut(df["num_shares"], bins=cut_points, labels=labels)
print(df['num_shares_Bin'].value_counts(), **sp)
