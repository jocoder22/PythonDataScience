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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
plt.style.use('ggplot')

from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
sp = '\n\n'

url = 'https://assets.datacamp.com/production/course_1939/datasets/gm_2008_region.csv'

colname = ['population', 'fertility', 'HIV', 'CO2', 'BMI_male',
           'GDP', 'BMI_female', 'life', 'child_mortality', 'Region']

df = pd.read_csv(url, sep=',')

print(df.head(), df.info(), sep=sp, end=sp)

df.loc[df['life'] < 62, 'lifeCat'] = 'veryLow'
df.loc[df['life'] >= 62, 'lifeCat'] = 'Low'
df.loc[df['life'] >= 72, 'lifeCat'] = 'Medium'
df.loc[df['life'] >= 76, 'lifeCat'] = 'High'

print(df.head(), df.info(), sep=sp, end=sp)
print(df['lifeCat'].value_counts())

y = df[['lifeCat']].values
X = df.drop(['life', 'Region', 'lifeCat'], axis=1).values

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=4)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
ypred = knn.predict(X_test)

accuracy = accuracy_score(y_test, ypred)
print(f'Accuracy score: {accuracy:.02f}')



'''
# Using odd number of estimators
# Use for classification problems only

# create individual estimators
k_neClas = KNeighborsClassifier(5)
d_treeClas = DecisionTreeClassifier()
logregClas = LogisticRegression()

# Create the voting classification model
voting_clf = VotingClassifier(
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

# Calculate accuracy
accuracy = accuracy_score(y_test, ypred)
print(f'Accuracy score: {accuracy:.02f}')

'''