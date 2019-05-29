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
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

colname = '''MPG Cylinders Displacement Horsepower Weight Acceleration 
             Model_year Origin Car_name'''.split()

origin_name = ['USA', 'Europe', 'Japan']


dataset = pd.read_csv(url, names=colname, na_values=['na','Na', '?'],
                skipinitialspace=True, comment='\t', sep=" ", quotechar='"')



dataset.drop(columns='Car_name', inplace=True)
print(dataset.isna().sum())