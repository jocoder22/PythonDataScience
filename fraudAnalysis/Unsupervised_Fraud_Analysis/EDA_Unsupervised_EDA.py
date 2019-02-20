#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
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

sp = '\n\n'

url2 = 'https://assets.datacamp.com/production/repositories/2162/datasets/08cfcd4158b3a758e72e9bd077a9e44fec9f773b/chapter_3.zip'

# download all the zip files
response = requests.get(url2)

# unzip the content
zipp = ZipFile(BytesIO(response.content))

# Dsiplay files names in the zip file
mylist = [filename for filename in zipp.namelist()]

print(mylist)
# Load data to DataFrame from file_path:
data = pd.read_csv(zipp.open(mylist[3]))
data = data.iloc[:, 1:]


print(data.groupby('category').mean(), data.head(), data.info(), sep='\n\n')

# Group by age groups and get the mean
print(data.groupby('age').mean(), end=sp)


# Count the values of the observations in each age group
print(data['age'].value_counts(), end=sp)

# Create two dataframes with fraud and non-fraud data
frauds = data[data.fraud == 1]
nonfraud = data[data.fraud == 0]

# Plot histograms of the amounts in fraud and non-fraud data
plt.hist(frauds.amount, alpha=0.5, label='fraud')
plt.hist(nonfraud.amount, alpha=0.5, label='nonfraud')
plt.xlabel('Amount in dollars($)')
plt.ylabel('Number of cases')
plt.legend()
plt.show()
