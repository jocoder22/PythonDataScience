#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
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
from sklearn.preprocessing import MinMaxScaler
plt.style.use('ggplot')

from sklearn.ensemble import votingClassifier

# Using odd number of estimators
# Use for classification problems only

# create individual estimators
k_neClas = KNeighborsClassifier(5)
d_treeClas = DecisionTreeClassifier()
logregClas = LogisticRegression()

# Create the voting classification model
voting_clf = votingClassifier(
    estimators = [
        ('k_ne', k_neClas),
        ('d_tree', d_treeClas),
        ('logreg', logregClas)
    ]
)

# fit the model
voting_clf.fit(X_train, y_train)

# Predict with the model
ypred = voting_clf.predict(X_test)