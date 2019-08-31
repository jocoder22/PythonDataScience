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

sp = {'sep':'\n\n', 'end':'\n\n'}
pp = '#'*80

mypath = r'C:\Users\Jose\Desktop\PythonDataScience\ensemble'


with changepath(mypath):
    df = pd.read_csv('lifeExp.csv')

print(df.head(), df.shape, **sp)

y = df['lifecat'].values
X = df.drop(['life','lifeCat', 'lifecat'], axis=1).values

# print(y.shape,y.head(), X.head(), X.shape, sep=sp, end=sp)
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1,
                                    stratify=y)


# scale the dataset
scaler = StandardScaler()

# First fit on training set only.
scaler.fit(X)

# then transform to both the training set and the test set.
XtrainScaled = scaler.transform(X_train)
XtestScaled= scaler.transform(X_test)


# Best kneighbors model
k_neClas = KNeighborsClassifier(7)
# Fit the classifier to the training data
k_neClas.fit(XtrainScaled, y_train)
kscore = k_neClas.score(XtestScaled, y_test)
print(f'Accuracy score for KNeighhors Classifier: {kscore:.02f}', **sp)

# Best parameters for decisison Tree
bdtp = {'criterion':'gini', 'max_depth': 4, 'max_features': 'auto', 'min_samples_split': 2}
bdtp2 = {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'auto', 'min_samples_split': 5}
best_dt = DecisionTreeClassifier(**bdtp, random_state=5)
best_dt.fit(XtrainScaled, y_train)
dscore = best_dt.score(XtestScaled, y_test)
print(f'Accuracy score for Decision Tree Classifier: {dscore:.02f}', **sp)

# best model parameters for RandomForest model
rfp = {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 5, 'max_features': 'auto', 'min_samples_split': 3, 'n_estimators': 27}
rfp2 = {'bootstrap': False, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto', 'min_samples_split': 3, 'n_estimators': 28}
best_rf = RandomForestClassifier(**rfp, random_state=5)
best_rf.fit(XtrainScaled, y_train)
fscore = best_rf.score(XtestScaled, y_test)
print(f'Accuracy score for RandomForest Classifier: {fscore:.02f}', **sp)


# logistic regression
logReg = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Fit the classifier to the training data
logReg.fit(XtrainScaled, y_train)

# Predict the labels of the test set: y_pred
y_pred = logReg.predict(XtestScaled)

lscore = logReg.score(XtestScaled, y_test)
print(f'Accuracy score for logistic Regression Classifier: {lscore:.02f}', **sp)

# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
print(f'Score test accuracy: {logReg.score(XtestScaled, y_test):.02f}', **sp)



# Create and fit the voting classifier
clf_vote = VotingClassifier(
    estimators=[('knn', k_neClas), ('dt', best_dt), ('lr', logReg)],
    voting='soft',
    weights=[1,1,2]
)
clf_vote.fit(XtrainScaled, y_train)
vscore = clf_vote.score(XtestScaled, y_test)
print(f'Accuracy score for voting Classifier: {vscore:.02f}', **sp)
