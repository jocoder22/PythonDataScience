#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests 

import seaborn as sns

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.cluster.vq import vq, kmeans, whiten

# plt.style.use('ggplot')


path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = '\n\n'

# Read in the dataset, 
data = pd.read_csv('car.csv', compression='gzip')

origin = data.pop('Origin')
modelyear = data.pop('Model_year')

# Normalize the dataset
data2 = whiten(data)
dataset2 = pd.DataFrame(data2, columns=data.columns)

# Cluster using kmeans
k_cluster, _ = kmeans(dataset2, 3)
dataset2['k_labels'], _ = vq(dataset2, k_cluster)

print(dataset2.groupby('k_labels').mean(), end=sp)
print(dataset2.groupby('k_labels').count(), end=sp)
print(dataset2.head())