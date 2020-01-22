import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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