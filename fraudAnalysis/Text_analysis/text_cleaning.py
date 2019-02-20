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
