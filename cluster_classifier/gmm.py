#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM

from printdescribe import print2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
print(__doc__)

RANDOMSTATE = 42

# download the data
iris = datasets.load_iris()

features = iris.data
targets = iris.target
labelnames = iris.target_names


# create train/test split using 25% test size
X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                    test_size=0.30,
                                                    random_state=RANDOMSTATE)  


n_classes = len(np.unique(y_train))


# Try GMMs.
classifier =  GMM(n_components=n_classes, covariance_type= 'tied')



# define colors and markers
markers = ["*","o", "+"]
colors = ["r", "y", "k"]
col = ['r*','yo','k+']
labels = y_test

# Fit to data and predict using pipelined scaling, PCA.
gmm = make_pipeline(StandardScaler(), classifier)
gmm.fit(X_train)
pca_result = pca.transform(X_test)

# plot
for i, label in enumerate(np.unique(y_test)):
    plt.plot(pca_result[:,0][labels==label],pca_result[:,1][labels==label], 
                col[int(i)], label=labelnames[i])
plt.title("The projection onto 2 PCA components")
plt.legend(loc='best', shadow=False, scatterpoints=3)
plt.show();


# Fit to data and predict using pipelined scaling, LDA.
lda = LDA(n_components=2)
lda.fit(X_train,y_train)
lda_result = lda.transform(X_test)

# plot
for i, label in enumerate(np.unique(y_test)):
    plt.scatter(lda_result[:,0][y_test==label],lda_result[:,1][y_test==label], 
                c=colors[i], marker=markers[i], lw=1, label=labelnames[i])
plt.title("The projection onto 2 LDA components with scaling")
plt.legend(loc='best', shadow=False, scatterpoints=3)
plt.show();


# Fit to data and predict without using pipelined scaling, LDA.
lda2 = LDA(n_components=2)
lda2.fit(X_train,y_train)
lda_result = lda2.transform(X_test)

# plot
for i, label in enumerate(np.unique(y_test)):
    plt.scatter(lda_result[:,0][y_test==label],lda_result[:,1][y_test==label], 
                c=colors[i], marker=markers[i], lw=1, label=labelnames[i])
plt.title("The projection onto 2 LDA components without scaling")
plt.legend(loc='best', shadow=False, scatterpoints=3)
show;
