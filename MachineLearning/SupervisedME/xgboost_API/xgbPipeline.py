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


####################################################################################
####################################################################################
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


dataData = houseData["YearBuilt"].value_counts()
myy = houseData["YearBuilt"].unique()
print2(dataData, myy)
for key, value in dataData.items():
    print(key, value)
    if value >= 10:
        houseData.loc[houseData['YearBuilt'] == key, 'Catt'] = str(key)
    else:
        houseData.loc[houseData['YearBuilt'] == key, 'Catt'] =  "Cat" + str(value)
    
    
catvalues(houseData, "YearBuilt", "Catt")
print2(houseData.info(), houseData.dtypes, houseData.head(), houseData.shape)
print2(houseData[["YearBuilt", "Catt"]].head(20))
catvalues(houseData, "YearBuilt", "Catt")
datacat = houseData["Catt"].value_counts()
print2(datacat.head(35), datacat.tail(35))



print(cardata.head(), end=sp)
print(pd.unique(cardata['Model_year']), end=sp)
print(cardata['Model_year'].value_counts(), end=sp)
print2(houseData[["YearBuilt", "Catt"]].head(20))
                      
                      