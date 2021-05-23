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
from ctmanager.functionX import changepath

mypath = r'C:\Users\Jose\Desktop\PythonDataScience\ensemble'  
sp = '\n\n'
pp = '#'*80

url = 'https://assets.datacamp.com/production/course_1939/datasets/gm_2008_region.csv'
colname = ['population', 'fertility', 'HIV', 'CO2', 'BMI_male',
           'GDP', 'BMI_female', 'life', 'child_mortality', 'Region']

df = pd.read_csv(url, sep=',')
print(df.head(), df.info(), sep=sp, end=sp)

df.loc[df['life'] < 62, 'lifeCat'] = 'veryLow'
df.loc[df['life'] >= 62, 'lifeCat'] = 'Low'
df.loc[df['life'] >= 72, 'lifeCat'] = 'Medium'
df.loc[df['life'] >= 76, 'lifeCat'] = 'High'

df2 = df.copy()
print(df.head(), df.info(), sep=sp, end=sp)
print(df['lifeCat'].value_counts())

# mapper = {'veryLow':0, 'Low':1, 'Medium':2, 'High':3}
# df['lifeCat'] = df.lifeCat.map(mapper)
# # print(df['lifeCat2'].value_counts())
# Using sklearn
label1 = LabelEncoder()
hot1 = OneHotEncoder()


df['lifecat'] = label1.fit_transform(df['lifeCat'])
df['rc'] =  df.Region

# df['Rcoded'] = label1.fit_transform(df['Region'])
# Form dummies from the Region category, return new dataset with Region drop
df = pd.get_dummies(df, columns=['Region'], prefix='R') #drop_first=True

with changepath(mypath):
    df.to_csv('lifeExp.csv', index=False)


y = df['lifecat'].values
X = df.drop(['life','lifeCat', 'lifecat', 'rc'], axis=1).values

# scale the dataset
scaler = StandardScaler()

# First fit on training set only.
scaler.fit(X)

# then transform to both the training set and the test set.
# print(y.shape,y.head(), X.head(), X.shape, sep=sp, end=sp)
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

X_train = scaler.transform(X_train)
X_test= scaler.transform(X_test)

with changepath(mypath):
    np.savez('mydata', X_train=X_train, X_test=X_test, 
                y_train=y_train, y_test=y_test)

'''
with changepath(mypath):
    npzfile = np.load('mydata.npz')
    X_train, X_test =  npzfile['X_train'], npzfile['X_test']
    y_train, y_test =npzfile['y_train'], npzfile['y_test']


nc = 0

# search for k with highest accuracy
for n in range(2,20):
    # Instantiate a k-NN classifier: knn
    knn = KNeighborsClassifier(n_neighbors=n)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Predict the labels of the test data: y_pred
    ypred = knn.predict(X_test)

    
    accuracy = accuracy_score(y_test, ypred)
    if accuracy > nc:
        print(f'Accuracy score for {n} neighbors: {accuracy:.02f}')
        nc = accuracy

k_neClas = KNeighborsClassifier(7)


# fit the DecisionTreeClassifier
mytree3 = DecisionTreeClassifier(criterion='gini', max_depth=5, 
                            random_state=1)

mytree3.fit(X_train, y_train)

treescore = mytree3.score(X_test,y_test)
print(f'Accuracy score for DecisionTree Classifier: {treescore:.02f}')

print(f'{pp}', end=sp)
print(mytree3.get_params().keys())

"""
'criterion', 'max_depth', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
'min_impurity_split', 'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 
'presort', 'random_state', 'splitter'
"""

param_grid2 = {'max_features': ['auto', 'log2'],  'max_depth': [3, 4, 5, 7], 
                "min_samples_split": [2, 3, 5, 7, 10], 'criterion': ['gini', 'entropy']}


# Define the model to use
model_d = DecisionTreeClassifier(random_state=5)

# Combine the parameter sets with the defined model
CV_model2 = GridSearchCV(
    estimator=model_d, param_grid=param_grid2, cv=5, scoring='f1_micro', n_jobs=-1)

 # Fit the model to our training data and obtain best parameters
CV_model2.fit(X_train, y_train)
print('################      this is printing for Decision Tree Model       ###########')
print(CV_model2.best_params_, CV_model2.best_score_, sep=sp, end=sp)


# Best parameters for decisison Tree
bdtp = {'criterion':'gini', 'max_depth': 4, 'max_features': 'auto', 'min_samples_split': 2}

best_dt = DecisionTreeClassifier(bdtp, random_state=5)







# fit a randomforest tree model
forest = RandomForestClassifier(criterion='gini', n_estimators=50,
                                 random_state=1, n_jobs=2)

forest.fit(X_train, y_train)

fscore = forest.score(X_test,y_test)
print(f'Accuracy score for RandomForest Classifier: {fscore:.02f}')


# Do parameter search
# Define the parameter sets to test
param_grid = {'n_estimators': range(1,36), 'max_features': ['auto', 'log2'],  
                'max_depth': [3, 4, 5, 7], "bootstrap": [True, False], 
                "min_samples_split": [2, 3, 5, 7, 10], 'criterion': ['gini', 'entropy']}

# Define the model to use
model7 = RandomForestClassifier(random_state=5)
print(model7.get_params().keys())

# Combine the parameter sets with the defined model
CV_model = GridSearchCV(
    estimator=model7, param_grid=param_grid, cv=5, scoring='f1_micro', n_jobs=-1)

 # Fit the model to our training data and obtain best parameters
CV_model.fit(X_train, y_train)
print('################     this is printing for RandomForest Model     ###########')
print(CV_model.best_params_, CV_model.best_score_, sep=sp, end=sp)


# best model parameters for RandomForest model
rfp = {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 5, 'max_features': 'auto', 'min_samples_split': 3, 'n_estimators': 27}

best_rf = RandomForestClassifier(rfp, random_state=5)



# Add the logistic model
# Create the classifier: logreg
logReg = LogisticRegression(multi_class='multinomial', solver = 'lbfgs')

# scale the dataset
scaler = StandardScaler()

# First fit on training set only.
scaler.fit(X_train)

# then transform to both the training set and the test set.
XtrainScaled = scaler.transform(X_train)
XtestScaled = scaler.transform(X_test)

# Fit the classifier to the training data
logReg.fit(XtrainScaled, y_train)

# Predict the labels of the test set: y_pred
y_pred = logReg.predict(XtestScaled)


# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))



#################################################################################################
#################################################################################################
# using the wine dataset
print(f'{pp}', end=sp)
wine = datasets.load_wine()

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

nc = 0

# search for k with highest accuracy
for n in range(2,36):
    # Instantiate a k-NN classifier: knn
    knn2 = KNeighborsClassifier(n_neighbors=n)

    # Fit the classifier to the training data
    knn2.fit(X_train, y_train)

    # Predict the labels of the test data: y_pred
    y_pred = knn2.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > nc:
        print(f'Accuracy score for {n} neighbors: {accuracy:.02f}')
        nc = accuracy


'''