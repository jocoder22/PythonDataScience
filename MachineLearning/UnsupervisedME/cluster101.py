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

# path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
path = r"D:\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)

sp = '\n\n'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# urlname = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names'
# urlindex = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/Index"

url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
urlname = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names"
urlindex = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/Index"

for i in [urlname, urlindex, url2]:
    respond = requests.get(i)
    text = respond.text
    print(text, end=sp)

colname = '''MPG Cylinders Displacement Horsepower Weight Acceleration 
             Model_year Origin Car_name'''.split()

dataset = pd.read_csv(url, names=colname, na_values=['na','Na', '?'],
                skipinitialspace=True, comment='\t', sep=" ", quotechar='"')


dataset.drop(columns='Car_name', inplace=True)
dataset.dropna(inplace=True)

dataset.to_csv('car.csv', index=False, compression='gzip')

# data3 = dataset.loc[:, ['MPG', 'Horsepower']]
# print(data3.head())

dataset2 = whiten(dataset)
dataset2 = pd.DataFrame(dataset2, columns=colname[:8])

# plot the original data
plt.plot(dataset['MPG'], label='original')

# plot the scaled data
plt.plot(dataset2['MPG'], label='scaled')

plt.legend()
plt.show()





# Hierarchical clustering
# Number of cluster investigation
distance = linkage(dataset2[['MPG', 'Horsepower']], method='ward')
dnn = dendrogram(distance)
plt.show()

# define the cluster labels
dataset2['labels'] = fcluster(distance, 3, criterion='maxclust')

# plot the clusters
sns.scatterplot(x='MPG', y='Horsepower', 
                hue='labels', data=dataset2)
plt.show()


# Kmeans clustering
# Define the cluster centers
K_cluster, _ = kmeans(dataset2[['MPG', 'Horsepower']], 4)

# Define the cluster labels
dataset2['k_labels'], _ = vq(dataset2[['MPG', 'Horsepower']], K_cluster)

# plot the clusters
sns.scatterplot(x='MPG', y='Horsepower', 
                hue='k_labels', data=dataset2)
plt.show()


# finding number of clusters
distortions = []
n_clusters = range(1, 23)

# Create a list of distortions 
for idx in n_clusters:
    _ , distortion = kmeans(dataset2[['MPG', 'Horsepower']], idx)
    distortions.append(distortion)

# Create a data frame with two lists
_plot = pd.DataFrame({'n_clusters': n_clusters, 'distortions': distortions})


# Create a line plot of num_clusters and distortions
sns.lineplot(x='n_clusters', y='distortions', data=_plot)
plt.show()


# Kmeans clustering
# Define the cluster centers
K_cluster, _ = kmeans(dataset2, 4)

# Define the cluster labels
dataset2['k_labels'], _ = vq(dataset2, K_cluster)

# plot the clusters
sns.scatterplot(x='MPG', y='Horsepower', 
                hue='k_labels', data=dataset2)
plt.show()



# finding number of clusters
distortions = []
n_clusters = range(1, 23)

# Create a list of distortions 
for idx in n_clusters:
    _ , distortion = kmeans(dataset2, idx)
    distortions.append(distortion)

# Create a data frame with two lists
_plot = pd.DataFrame({'n_clusters': n_clusters, 'distortions': distortions})


# Create a line plot of num_clusters and distortions
sns.lineplot(x='n_clusters', y='distortions', data=_plot)
plt.show()
