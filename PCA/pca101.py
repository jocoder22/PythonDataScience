#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

sp = '\n\n'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'


colname = '''MPG Cylinders Displacement Horsepower Weight Acceleration 
             Model_year Origin Car_name'''.split()

dataset = pd.read_csv(url, names=colname, na_values=['na','Na', '?'],
                skipinitialspace=True, comment='\t', sep=" ", quotechar='"')

dataset.drop(columns='Car_name', inplace=True)
print(dataset.isna().sum())
dataset.dropna(inplace=True) 


scaler =  StandardScaler()
feature = dataset.pop('MPG')
dataset = scaler.fit_transform(dataset)


pca = PCA()
pcomp = pca.fit_transform(dataset)
pcompf= pca.fit(dataset)
pcomp_df = pd.DataFrame(pcomp, columns=colname[1:8])

# Create a pairplot
sns.pairplot(pcomp_df)
plt.show()

# Inspect the explained variance ratio per component
print(pcompf.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())