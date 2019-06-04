#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import seaborn as sns

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import vq, kmeans

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

print(dataset.head(), dataset.shape, sep=sp, end=sp)