import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import datasets

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")
        
def catvalues(ddtt, *args):
    for arg in  args:
        print(ddtt[arg].value_counts(), end="\n\n")
        
def catvalues2(ddtt):
    for col in ddtt.columns:
        if ddtt[col].dtypes == 'object':
            print(ddtt[col].value_counts(), end="\n\n")
            
sp = "\n\n"
mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
mytt = r"D:\PythonDataScience\MachineLearning\UnsupervisedME"

# load the dataset, 
cardata = pd.read_csv(os.path.join(mytt, "car.csv"), compression='gzip')





# Load the California Housing Data     
calhousing = datasets.fetch_california_housing()
X = pd.DataFrame(calhousing.data, columns=calhousing.feature_names)
y = calhousing.target

print2(X[:6], y[:6], X.dtypes)

# load pickle file
onlinedata = pd.read_pickle(os.path.join(mydir, "onlinedata.pkl"))
print2(onlinedata.head(), onlinedata.info(), y[:10])

newpath = "D:\PythonDataScience\MachineLearning\FeatureEngineering"
houseData = pd.read_csv(os.path.join(newpath, "housing.csv"))
print2(houseData.info(), houseData.dtypes, houseData.head(), houseData.shape)

catvalues(houseData, "RoofStyle", "CentralAir", "YearBuilt")
catvalues2(houseData)
