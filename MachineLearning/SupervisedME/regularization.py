import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")
        
        
# Load the load_breast_cancer dataset
breast_X, breast_y = datasets.load_breast_cancer(return_X_y=True)

print2(breast_X, breast_y, breast_X[0].size)
