#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import datasets
# from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from sklearn import mixture

from printdescribe import print2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
print(__doc__)

RANDOMSTATE = 42
show = plt.show()

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

# # Break up the dataset into non-overlapping training (80%) and testing (20%) sets
# skf = StratifiedKFold(n_splits=5)

# # Only take the first fold.
# train_index, test_index = next(iter(skf.split(iris.data, iris.target)))

# X_train = iris.data[train_index]
# y_train = iris.target[train_index]
# X_test = iris.data[test_index]
# y_test = iris.target[test_index]


# Try GMMs.
classifier =  mixture.GaussianMixture(n_components=n_classes, covariance_type= 'tied')
classifier.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

# define colors and markers
markers = ["*","o", "+"]
# colors = ["r", "y", "k"]
colors = ['navy', 'turquoise', 'darkorange']
col = ['r*','yo','k+']
labels = y_test

# Fit to data and predict using pipelined scaling, PCA.
gmm = make_pipeline(StandardScaler(), classifier)
gmm.fit(X_train)
pred_train = gmm.predict(X_train)
train_accuracy = np.mean(pred_train.ravel() == y_train.ravel()) * 100
print2(f"Train accuracy: {np.round(train_accuracy,2)}")
  
pred_test = gmm.predict(X_test)
test_accuracy = np.mean(pred_test.ravel() == y_test.ravel()) * 100
print2(f"Test accuracy: {np.round(test_accuracy,2)}")


 for n, color in enumerate(colors):
  # plot the original data
    data = iris.data[iris.target == n]
    plt.scatter(data[:, 0], data[:, 1], s=2.8, color=color,label=iris.target_names[n])
    
  # plot the test data
    data = X_test[y_test == n]
    plt.plot(data[:, 0], data[:, 1], 'x', color=color)
    
  # plot the prediction
    data = X_test[pred_test == n]
    plt.scatter(data[:, 0], data[:, 1], marker='o', color="white", edgecolor=color)
    
 show;
    
    
# # plot
# for i, label in enumerate(np.unique(y_test)):
#     plt.plot(pca_result[:,0][labels==label],pca_result[:,1][labels==label], 
#                 col[int(i)], label=labelnames[i])
# plt.title("The projection onto 2 PCA components")
# plt.legend(loc='best', shadow=False, scatterpoints=3)
# plt.show();


# # Fit to data and predict using pipelined scaling, LDA.
# lda = LDA(n_components=2)
# lda.fit(X_train,y_train)
# lda_result = lda.transform(X_test)

# # plot
# for i, label in enumerate(np.unique(y_test)):
#     plt.scatter(lda_result[:,0][y_test==label],lda_result[:,1][y_test==label], 
#                 c=colors[i], marker=markers[i], lw=1, label=labelnames[i])
# plt.title("The projection onto 2 LDA components with scaling")
# plt.legend(loc='best', shadow=False, scatterpoints=3)
# plt.show();


# # Fit to data and predict without using pipelined scaling, LDA.
# lda2 = LDA(n_components=2)
# lda2.fit(X_train,y_train)
# lda_result = lda2.transform(X_test)

# # plot
# for i, label in enumerate(np.unique(y_test)):
#     plt.scatter(lda_result[:,0][y_test==label],lda_result[:,1][y_test==label], 
#                 c=colors[i], marker=markers[i], lw=1, label=labelnames[i])
# plt.title("The projection onto 2 LDA components without scaling")
# plt.legend(loc='best', shadow=False, scatterpoints=3)
# show;



"""

colors = ['navy', 'turquoise', 'darkorange']


def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')

iris = datasets.load_iris()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(iris.data, iris.target)))


X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = {cov_type: GaussianMixture(n_components=n_classes,
              covariance_type=cov_type, max_iter=20, random_state=0)
              for cov_type in ['spherical', 'diag', 'tied', 'full']}

n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)

    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
                    label=iris.target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))


plt.show()

"""
