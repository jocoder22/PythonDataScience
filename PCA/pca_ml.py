#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# Import different modules for using with the notebook
from IPython.display import display
from IPython.display import Image
from IPython.display import HTML

from printdescribe import print2

# download the data
iris = datasets.load_iris()

data = iris.data
labels = iris.target
labelnames = iris.target_names

# Fit PCA
pca = PCA(n_components=3)
pca.fit(data)

# Plot
plt.plot(range(0, 3), pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Principal Components')
plt.title('Explained Variance Ratio')
plt.show()
