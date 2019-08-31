#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
plt.style.use('ggplot')

from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from contextlib import contextmanager
from functionX import changepath

sp = '\n\n'
pp = '#'*80

mypath = r'C:\Users\Jose\Desktop\PythonDataScience\ensemble'


with changepath(mypath):
    df3 = pd.read_csv('lifeExp.csv')

print(df3.head(), end=sp)


# Best kneighbors model
k_neClas = KNeighborsClassifier(12)

# Best parameters for decisison Tree
bdtp = {'criterion':'gini', 'max_depth': 4, 'max_features': 'auto', 'min_samples_split': 2}
best_dt = DecisionTreeClassifier(bdtp, random_state=5)

# best model parameters for RandomForest model
rfp = {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 5, 'max_features': 'auto', 'min_samples_split': 3, 'n_estimators': 27}
best_rf = RandomForestClassifier(rfp, random_state=5)