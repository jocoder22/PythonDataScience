#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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



# Using Pipeline
pipe1 = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=4))])

# Fit it to the dataset and extract the component vectors
pipe1.fit(dataset)
vectors = pipe1.steps[1][1].components_.round(2)


# Print feature effects
print('PC 1 effects = ' + str(dict(zip(colname[1:8], vectors[0]))))
print('PC 2 effects = ' + str(dict(zip(colname[1:8], vectors[1]))))
print('PC 3 effects = ' + str(dict(zip(colname[1:8], vectors[2]))))
print('PC 4 effects = ' + str(dict(zip(colname[1:8], vectors[3]))))
