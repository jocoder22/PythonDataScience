#!/usr/bin/env python
import os
import pickle
from functools import reduce
from operator import mul

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.regression.linear_model import OLS
from sklearn import linear_model
from sklearn.decomposition import PCA

import holoviews as hv
import hvplot
import hvplot.pandas

np.random.seed(42)
hv.extension('bokeh')

pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader.famafrench import get_available_datasets
import pandas_datareader.data as web

from printdescribe import print2

# Download datasets
portfolios100 = web.DataReader('100_Portfolios_10x10_Daily', 'famafrench')
factors5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_Daily', 'famafrench')

print(portfolios100['DESCR'])
print(factors5['DESCR'])

# select the Average Equal Weighted Returns -- Daily (1220 rows x 100 cols)
portfolios = portfolios100[1]
factors = factors5[0]

# pd.melt(portfolios.head(100).divide(100).add(1).cumprod().reset_index(), 
#         id_vars='Date').plot.line(x='Date', by='variable')
pp = portfolios.iloc[:,0:12].divide(100).add(1).cumprod()
# pp.plot()
# plt.show()

portfolios = portfolios.divide(100)

# pd.melt(factors[0].head(100).divide(100).add(1).cumprod().reset_index(), 
#         id_vars='Date').hvplot.line(x='Date', by='variable')

# ff = factors.head(100).divide(100).add(1).cumprod()
# ff.plot()
# plt.show()


# factors = factors.loc[portfolios.index,:].divide(100)
# hvplot.scatter_matrix(factors)

# Create a pairplot
# sns.pairplot(factors)
# plt.show()


pca_factors = PCA()
pca_factors.fit(factors.dropna())

# pd.Series(pca_factors.explained_variance_ratio_,name='Variance_Explained').hvplot.line(
#     label='Scree Plot of PCA Variance Explaned (%)').redim(Variance_Explained={'range': (0, 1)})

pe = pd.Series(pca_factors.explained_variance_ratio_,name='Variance_Explained')
print2(pe)
plt.plot(pe.index, pe.values)
plt.title('Scree Plot of PCA Variance Explaned (%)')
plt.show()


tt = pd.DataFrame(pe.values.cumsum())
plt.plot(tt)
plt.title('Scree Plot of PCA Variance Explaned (%)')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.grid()
plt.show();

print(pca_factors.explained_variance_ratio_.cumsum())

# We must make sure we have an overlapping dataset
dates = np.intersect1d(factors.index, portfolios.index)
factors = factors.loc[dates,:]
portfolios = portfolios.loc[dates,:]

factors = factors.loc[~factors.isna().any(1)&~portfolios.isna().any(1),:]
portfolios = portfolios.loc[~factors.isna().any(1)&~portfolios.isna().any(1),:]

lm = linear_model.LinearRegression(normalize=True)
lm.fit(X=factors, y=portfolios)
