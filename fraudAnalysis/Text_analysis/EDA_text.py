#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import seaborn as sns
plt.style.use('ggplot')


sp = '\n\n'

url2 = 'https://assets.datacamp.com/production/repositories/2162/datasets/94f2356652dc9ea8f0654b5e9c29645115b6e77f/chapter_4.zip'

# download all the zip files
response = requests.get(url2)

# unzip the content
zipp = ZipFile(BytesIO(response.content))

# Dsiplay files names in the zip file
mylist = [filename for filename in zipp.namelist()]


data = pd.read_csv(zipp.open(mylist[5]))
print(mylist, data.head(), data.columns, sep=sp)

print(data[['content', 'clean_content']].head())


mask = data['clean_content'].str.contains('sell enron stock', na=False)

# Find all cleaned emails that contain 'sell enron stock'
mask = data['clean_content'].str.contains('sell enron stock', na=False)

# Select the data from data using the mask
print(data.loc[mask])


# Create a list of terms to search for
searchfor = ['enron stock', 'sell stock', 'stock bonus', 'sell enron stock']

# Filter cleaned emails on searchfor list and select from data
filtered_emails = data.loc[data['clean_content'].str.contains(
    '|'.join(searchfor), na=False)]

print(filtered_emails.head())


# Create flag variable where the emails match the searchfor terms
data['flag'] = np.where(
    (data['clean_content'].str.contains('|'.join(searchfor)) == True), 1, 0)

# Count the values of the flag variable
count = data['flag'].value_counts()
print(count, data.head(), data.columns, sep=sp)
