#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
from EDA import eda
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


url = 'https://assets.datacamp.com/production/repositories/2162/datasets/4fb6199be9b89626dcd6b36c235cbf60cf4c1631/chapter_2.zip'

# download all the zip files
response = requests.get(url)

# unzip the content
zipp = ZipFile(BytesIO(response.content))

# Dsiplay files names in the zip file
print(zipp.namelist())

mylist = [filename for filename in zipp.namelist()]

print(mylist)

# Load data to DataFrame from file_path: 
data = pd.read_csv(zipp.open(mylist[1]))
data['Class'] = data.Class.astype('category')

# print(data.Class.value_counts())
# eda(data)

# Calculate the percentage of non fraud observation
print(data.Class.value_counts(normalize=True) * 100)

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
# Split your data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


# Define the model as the random forest
model = RandomForestClassifier(random_state=5)
# Fit the model to our training set
model.fit(X_train, y_train)

# Obtain predictions from the test data
predicted = model.predict(X_test)


