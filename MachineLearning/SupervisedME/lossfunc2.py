import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")
        
        
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

print2(diabetes_X, diabetes_y)