#!/usr/bin/env python
from sklearn.ensemble import VotingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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


def getresults(X_train, y_train, X_test, y_test, model):
  model.fit(X_train, y_train)
  predicted = model.predict(X_test)
  probs = model.predict_proba(X_test)
  print(roc_auc_score(y_test, probs[:, 1]))
  print(classification_report(y_test, predicted))
  print(confusion_matrix(y_test, predicted))


def getresults2(X_train, y_train, X_test, y_test, model):
  model.fit(X_train, y_train)
  predicted = model.predict(X_test)
  print(classification_report(y_test, predicted))
  print(confusion_matrix(y_test, predicted))

url = 'https://assets.datacamp.com/production/repositories/2162/datasets/4fb6199be9b89626dcd6b36c235cbf60cf4c1631/chapter_2.zip'

# download all the zip files
response = requests.get(url)

# unzip the content
zipp = ZipFile(BytesIO(response.content))

# Dsiplay files names in the zip file
mylist = [filename for filename in zipp.namelist()]

# Load data to DataFrame from file_path:
data = pd.read_csv(zipp.open(mylist[1]))


X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# Split your data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)


model = LogisticRegression(class_weight={0: 1, 1: 15}, random_state=5)

# Get the model results
getresults(X_train, y_train, X_test, y_test, model)


# Define the three classifiers to use in the ensemble
clf1 = LogisticRegression(class_weight='balanced', random_state=5)
clf2 = RandomForestClassifier(class_weight='balanced_subsample', criterion='gini', max_depth=8, max_features='log2',
                              min_samples_leaf=10, n_estimators=30, n_jobs=-1, random_state=5)
clf3 = DecisionTreeClassifier(random_state=5, class_weight="balanced")
clf4 = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy',
                              max_depth=8, max_features='auto',  min_samples_leaf=10, n_estimators=30, n_jobs=-1, random_state=8)

# Combine the classifiers in the ensemble model
ensemble_model = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3), ('rf2', clf4)], voting='hard')

# Get the results
getresults2(X_train, y_train, X_test, y_test, ensemble_model)


# Split your data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


# Define the three classifiers to use in the ensemble
clf11 = LogisticRegression(class_weight={0: 1, 1: 15}, random_state=5)
clf21 = RandomForestClassifier(class_weight={0: 1, 1: 12}, criterion='gini', max_depth=8, max_features='log2',
                              min_samples_leaf=10, n_estimators=30, n_jobs=-1, random_state=5)
clf31 = DecisionTreeClassifier(random_state=5, class_weight='balanced')
clf41 = RandomForestClassifier(class_weight={0: 1, 1: 12}, criterion='entropy',
                              max_depth=18, max_features='auto',  min_samples_leaf=10, n_estimators=30, n_jobs=-1, random_state=5)

# Combine the classifiers in the ensemble model
ensemble_model = VotingClassifier(
    estimators=[('lr', clf11), ('rf', clf21), ('dt', clf31),('rf2', clf41)], voting='hard')

# Get the results
getresults2(X_train, y_train, X_test, y_test, ensemble_model)


# Define the three classifiers to use in the ensemble
clf12 = LogisticRegression(class_weight={0: 1, 1:15}, random_state=5)
clf22 = RandomForestClassifier(class_weight={0: 1, 1:12}, criterion='gini', max_depth=8, max_features='log2',
                              min_samples_leaf=10, n_estimators=30, n_jobs=-1, random_state=5)
clf32 = DecisionTreeClassifier(random_state=5, class_weight="balanced")

# Combine the classifiers in the ensemble model
ensemble_model = VotingClassifier(
    estimators=[('lr', clf12), ('rf', clf22), ('dt', clf32)], voting='hard')

# Get the results
getresults2(X_train, y_train, X_test, y_test, ensemble_model)


# Define the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf12), ('rf', clf22), (
    'gnb', clf32)], voting='soft', weights=[1, 4, 1], flatten_transform=True)

# Get results
getresults2(X_train, y_train, X_test, y_test, ensemble_model)

