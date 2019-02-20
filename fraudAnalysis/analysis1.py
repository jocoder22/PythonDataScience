#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
from EDA import eda
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\fraudAnalysis\\'
os.chdir(path)
url = 'https://assets.datacamp.com/production/repositories/2162/datasets/4fb6199be9b89626dcd6b36c235cbf60cf4c1631/chapter_2.zip'

# download all the zip files
response = requests.get(url)

# unzip the content
zipp = ZipFile(BytesIO(response.content))

# Dsiplay files names in the zip file
print(zipp.namelist())

mylist = [filename for filename in zipp.namelist()]

print(mylist)

# Load data to DataFrame from file_path: 
data = pd.read_csv(zipp.open(mylist[1]))
data['Class'] = data.Class.astype('category')

# print(data.Class.value_counts())
# eda(data)

# Calculate the percentage of non fraud observation
print(data.Class.value_counts(normalize=True) * 100)

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
# Split your data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


# Define the model as the random forest
model = RandomForestClassifier(random_state=5)
# Fit the model to our training set
model.fit(X_train, y_train)

# Obtain predictions from the test data
predicted = model.predict(X_test)

# Fit the model to our training set
model.fit(X_train, y_train)

# Obtain predictions from the test data
predicted = model.predict(X_test)

# Predict probabilities
probs = model.predict_proba(X_test)


# Print the accuracy performance metric
print(accuracy_score(y_test, predicted))

average_precision = average_precision_score(y_test, predicted)

precision, recall, _ = precision_recall_curve(y_test, predicted)


# Print the ROC curve, classification report and confusion matrix
print(roc_auc_score(y_test, probs[:, 1]))
print(classification_report(y_test, predicted))
print(confusion_matrix(y_test, predicted))


def plotcurve(recall, precision, average_precision):
  plt.step(recall, precision, color='b', alpha=0.2, where='post')
  plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title(
      '2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
  plt.show()


plotcurve(recall, precision, average_precision)


# Define the model with balanced subsample
model = RandomForestClassifier(
    class_weight='balanced_subsample', random_state=5)

# Fit your training model to your training set
model.fit(X_train, y_train)

# Obtain the predicted values and probabilities from the model
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)

# Print the roc_auc_score, the classification report and confusion matrix
print(roc_auc_score(y_test, probs[:, 1]))
print(classification_report(y_test, predicted))
print(confusion_matrix(y_test, predicted))


model = RandomForestClassifier(bootstrap=True, class_weight={0: 1, 1: 12}, criterion='entropy',
                               max_depth=10,min_samples_leaf=10,
                               n_estimators=20, n_jobs=-1, random_state=5)


def getresults(X_train, y_train, X_test, y_test, model):
  model.fit(X_train, y_train)
  predicted = model.predict(X_test)
  probs = model.predict_proba(X_test)
  print(classification_report(y_test, predicted))
  print(confusion_matrix(y_test, predicted))

# Run the function get_model_results
getresults(X_train, y_train, X_test, y_test, model)


# Define the parameter sets to test
param_grid = {'n_estimators': [1, 30], 'max_features': ['auto', 'log2'],  'max_depth': [4, 8], 
                'criterion': ['gini', 'entropy']}

# Define the model to use
model7 = RandomForestClassifier(random_state=5)

# # Combine the parameter sets with the defined model
# CV_model = GridSearchCV(
#     estimator=model7, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)

#  # Fit the model to our training data and obtain best parameters
# CV_model.fit(X_train, y_train)
# CV_model.best_params_


# if __name__ == '__main__':
#     # Fit the model to our training data and obtain best parameters
#     CV_model.fit(X_train, y_train)
#     CV_model.best_params_


# {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'auto', 'n_estimators': 30}


# {'criterion': 'gini',
#  'max_depth': 8,
#  'max_features': 'log2',
#  'n_estimators': 30}


# Input the optimal parameters in the model
model = RandomForestClassifier(class_weight={0: 1, 1: 12}, criterion='gini',
                               max_depth=8, max_features='log2',  min_samples_leaf=10, n_estimators=30, n_jobs=-1, random_state=5)

# Get results from your model
getresults(X_train, y_train, X_test, y_test, model)


# Input the optimal parameters in the model
model = RandomForestClassifier(class_weight={0: 1, 1: 12}, criterion='entropy',
                               max_depth=8, max_features='auto',  min_samples_leaf=10, n_estimators=30, n_jobs=-1, random_state=5)

# Get results from your model
getresults(X_train, y_train, X_test, y_test, model)
