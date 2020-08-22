# Import different modules for using with the notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from printdescribe import print2
plt.rcParams["figure.figsize"] = 8,6

# Load in the `digits` data
digits = datasets.load_digits()
print2(digits.keys())

# Find the number of unique labels
number_digits = len(np.unique(digits.target))
print2(number_digits)

# Create a regular PCA model 
pca = PCA(n_components=2).fit(digits.data)

# Fit and transform the data to the model
reduced_data_pca = pca.transform(digits.data)

# Don't change the code in this block
colors = ['black', 'blue', 'purple', 'yellow', 'white',
          'red', 'lime', 'cyan', 'orange', 'gray']

# plt.figure(figsize=[12, 7])
for i in range(len(colors)):
    x = reduced_data_pca[:, 0][digits.target == i]
    y = reduced_data_pca[:, 1][digits.target == i]
    plt.scatter(x, y, marker='o', s=20, facecolors=colors[i], edgecolors='k')
#     plt.scatter(x, y, marker='o', s=55, facecolors=colors[i], edgecolors='k')
    
# PCA Scatter Plot
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("Regular PCA Scatter Plot")
plt.show()

