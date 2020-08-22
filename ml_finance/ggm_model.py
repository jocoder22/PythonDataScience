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


