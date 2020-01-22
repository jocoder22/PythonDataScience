import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")
        
        
# Load the load_breast_cancer dataset
breast_X, breast_y = datasets.load_breast_cancer(return_X_y=True)

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(
    breast_X, breast_y, test_size=0.2, random_state=42)

# We set random_state=0 for reproducibility 
sgdclassifier = SGDClassifier(max_iter=6000)


# List of loss function
loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']

# Instantiate the GridSearchCV object and run the search
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 
             'loss': loss, 'penalty':['l1','l2']}


# Fit gridsearch 
search = GridSearchCV(sgdclassifier, parameters, cv=6)
search.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print2("Best CV params", search.best_params_)
print2("Best CV accuracy", search.best_score_)
print2("Test accuracy of best grid search hypers:", search.score(X_test, y_test))