#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests 
# from sklearn import datasets
# from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
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
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
urlname = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names'
urlindex = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/Index"


# for i in [urlname, urlindex]:
#     respond = requests.get(i)
#     text = respond.text
#     print(text, end=sp)


colname = '''MPG Cylinders Displacement Horsepower Weight Acceleration 
             Model_year Origin Car_name'''.split()

origin_name = ['USA', 'Europe', 'Japan']

mytypes = ({cname: float if cname != 'Car_name' else str for cname in colname})

dataset = pd.read_csv(url, names=colname, dtype=mytypes, na_values=['na','Na', '?'],
                skipinitialspace=True, comment='\t', sep=" ", quotechar='"')

# print(dataset.head())

dataset.drop(columns='Car_name', inplace=True)
# print(dataset.isna().sum())
dataset.dropna(inplace=True)
dataset2 = dataset.copy()
# print(dataset.head(), dataset.shape, dataset.info(), sep=sp, end=sp)


encoder = OneHotEncoder(categorical_features = [7])
dataset = encoder.fit_transform(dataset).toarray()
newcolumns = colname[:-2]
origin_name.extend(newcolumns)
dataset = pd.DataFrame(dataset, columns=origin_name)

# print(dataset.head(), dataset.shape, dataset.info(), sep=sp)

originmap = {1.0 : 'USA', 2.0: 'Europe', 3.0: 'Japan'}
dataset2['Origin'] = dataset2['Origin'].map(originmap)
dataset2["Origin"] = dataset2["Origin"].astype('category')
data3 = pd.get_dummies(dataset2)
print(dataset2.head())
print(dataset2.info())

print(data3.head())
