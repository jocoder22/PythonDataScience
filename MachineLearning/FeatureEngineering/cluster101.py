#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import seaborn as sns

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import vq, kmeans, whiten

# plt.style.use('ggplot')


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

# data3 = dataset.loc[:, ['MPG', 'Horsepower']]
# print(data3.head())

dataset2 = whiten(dataset)
dataset2 = pd.DataFrame(dataset2, columns=colname[:8])
print(dataset.head(), dataset.shape, dataset2.head(), sep=sp, end=sp)



# Plot original data
plt.plot(dataset['MPG'], label='original')

# Plot scaled data
plt.plot(dataset2['MPG'], label='scaled')

plt.legend()
plt.show()