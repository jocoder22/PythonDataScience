#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Import different modules for using with the notebook
from IPython.display import display
from IPython.display import Image
from IPython.display import HTML

from printdescribe import print2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
print(__doc__)


RANDOMSTATE = 42
FIGSIZE = (12, 8)


features, targets = load_wine(return_X_y=True)

# create train/test split using 25% test size
X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                    test_size=0.30,
                                                    random_state=RANDOMSTATE)



# define colors and markers
markers = ["*","o", "+"]
colors = ["r", "y", "k"]

# Fit to data and predict using pipelined scaling, PCA.
pca = make_pipeline(StandardScaler(), PCA(n_components=2))
pca.fit(X_train)
pca_result = pca.transform(X_test)

# plot
for i, label in enumerate(np.unique(y_test)):
    plt.scatter(pca_result[:,0][labels==label],pca_result[:,1][labels==label], 
                c=colors[i], marker=markers[i], lw=1)
plt.title("The projection onto 2 PCA components");


# Fit to data and predict using pipelined scaling, LDA.
lda = LDA(n_components=2)
lda.fit(X_train,y_train)
lda_result = lda.transform(X_test)

# plot
for i, label in enumerate(np.unique(y_test)):
    plt.scatter(lda_result[:,0][y_test==label],lda_result[:,1][y_test==label], 
                c=colors[i], marker=markers[i], lw=1)
plt.title("The projection onto 2 LDA components with scaling");


# Fit to data and predict without using pipelined scaling, LDA.
lda2 = LDA(n_components=2)
lda2.fit(X_train,y_train)
lda_result = lda2.transform(X_test)

# plot
for i, label in enumerate(np.unique(y_test)):
    plt.scatter(lda_result[:,0][y_test==label],lda_result[:,1][y_test==label], 
                c=colors[i], marker=markers[i], lw=1)
plt.title("The projection onto 2 LDA components without scaling");
