#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests 

import seaborn as sns

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.cluster.vq import vq, kmeans, whiten
from printdescribe import changepath

# plt.style.use('ggplot')

# path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
path = r"E:\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = {"sep":"\n\n", "end":"\n\n"} 

# Read in the dataset, 
with changepath(path):
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


# Pivot table of clusters
dataset2['Origin'] = origin
dataset2['Model_year'] = modelyear
print(dataset2.groupby('k_labels')['Model_year', 'Origin'].count(), end=sp)
print(pd.pivot_table(dataset2, index='k_labels', columns='Model_year', fill_value=0,
            values="MPG", aggfunc='count', margins=True, margins_name='Total'), **sp)


groups = pd.crosstab(index=dataset2["k_labels"], 
                            columns=dataset2["Model_year"],
                             margins=True, margins_name='Total') # normalize=True).add_prefix('year_')
groups.index = ['Group_1', 'Group_2', 'Group_3', 'Total']
print(groups) 
