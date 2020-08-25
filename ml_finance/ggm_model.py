# Import different modules for using with the notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.random import randint

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM

from scipy.stats import multivariate_normal as mv_normal

from scipy.stats import mode 
import seaborn as sns 

from printdescribe import print2
plt.rcParams["figure.figsize"] = 8,6

rd42 = 42

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

# Create a regular LDA model 
lda = LDA(n_components=2).fit(digits.data, digits.target)

# Fit and transform the data to the model
reduced_data_lda = lda.transform(digits.data)

# Don't change the code in this block
colors = ['black', 'blue', 'purple', 'yellow', 'white',
          'red', 'lime', 'cyan', 'orange', 'gray']

for i in range(len(colors)):
    x = reduced_data_lda[:, 0][digits.target == i]
    y = reduced_data_lda[:, 1][digits.target == i]
    plt.scatter(x, y, marker='o', s=20, facecolors=colors[i], edgecolors='k')
    
# LDA Scatter Plot
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, scatterpoints=3,borderaxespad=0.)
plt.xlabel('First LDA Component')
plt.ylabel('Second LDA Component')
plt.title("LDA Scatter Plot")
plt.show()

# Split the `digits` data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, 
                digits.target, digits.images, test_size=0.25, random_state=rd42)

# Number of training features
n_samples, n_features = X_train.shape

# Print out `n_samples`
print('Number of samples:', n_samples)

# Print out `n_features`
print('Number of features:', n_features)

# Number of Training labels
n_digits = len(np.unique(y_train))
print ('Number of training labels:', n_digits)

# Inspect `y_train`
print('Number of labled data:', len(y_train))


# finding number of clusters using elbow method
distortions = []
n_clusters = range(1, 50)

for idx in n_clusters:
    kmeans = KMeans(n_clusters=idx, init='k-means++', random_state=42)
    kmeans.fit(X_train)
    distortions.append(kmeans.inertia_)

# Create a line plot of num_clusters and distortions
plt.plot(n_clusters, distortions, 'ro-')
plt.title('Number of Clusters: The Elbow Method ', size=15) 
plt.xlabel('Number of clusters', size=12)
plt.ylabel('Distortions', size=12)
plt.show();


# distortions percentage change
dt_series = pd.Series(distortions)

# calculate percentage change
dt_change = dt_series.pct_change()[1:]

# Create a line plot of num_clusters and percentage change in distortions
plt.plot(n_clusters[1:], dt_change, 'go-')
plt.title('Number of Clusters: The Elbow Method ', size=15) 
plt.xlabel('Number of clusters', size=12)
plt.ylabel('Distortions % change', size=12)
plt.grid()
plt.show();

# Create the KMeans model
# insert the code. Make sure you set init='k-means++', and random_state=42
ncluster = 10
kmean = KMeans(n_clusters=ncluster, init='k-means++', random_state=42)

# Fit the training data to the model
kmean.fit(X_train)

# Retrieve the cluster centres
centres =  kmean.cluster_centers_

# Don't change the code in this cell
# Figure size in inches
fig = plt.figure(figsize=(8, 3))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(centres[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')

# Show the plot
plt.show()


# Predict the labels for `X_test`
# Insert code
pred = kmean.predict(X_test)

# Print out the confusion matrix with `confusion_matrix()`
# insert code
mat = confusion_matrix(y_test, pred)
print2(mat)

# check for accuracy of the classification
accuracy_score(y_test, pred)

# display as a heatmap
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# random select the cluster
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X_train)
clusters = kmeans.predict(X_test)


labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(y_test[mask])[0]

mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

print2(mat)

# check for accuracy of the classification
accuracy_score(y_test, labels)

# Your code here
np.random.seed(42)

n_clusters, n_features = ncluster, X_test.shape[1]
init_array = np.zeros((n_clusters, n_features))

# random cetroid from each cluster
for i in range(n_clusters):
#     clusters = digits.data[digits.target==i]
    clusters = X_test[y_test==i]
    
    m,n = clusters.shape
    
    init_array[i, :] = clusters[randint(0, m-1), :]

# Your code here
# Create the KMeans model
# insert the code. Make sure you set init='k-means++', and random_state=42
kmean = KMeans(n_clusters=ncluster, init=init_array, n_init=1, random_state=42)

# Fit the training data to the model
kmean.fit(X_train)

# Retrieve the cluster centres
centres =  kmean.cluster_centers_

# Don't change the code in this cell
# Figure size in inches
fig = plt.figure(figsize=(8, 3))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(centres[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')

# Show the plot
plt.show()

# Predict the labels for `X_test`
# Insert code
pred = kmean.predict(X_test)


# Print out the confusion matrix with `confusion_matrix()`
# insert code
mat = confusion_matrix(y_test, pred)
print2(mat)

# display as a heatmap
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# check for accuracy of the classification
accuracy_score(y_test, pred)

clusters2 = pred

labels = np.zeros_like(clusters2)
for i in range(10):
    mask = (clusters2 == i)
    labels[mask] = mode(y_test[mask])[0]


mat2 = confusion_matrix(y_test, labels)


sns.heatmap(mat2.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# check for accuracy of the classification
accuracy_score(y_test, labels)


##########################################################################################
########################### GMM model ####################################################
##########################################################################################
data = X_train.data
# np.random.seed(1)

# Your code here
gmm_model = GMM(n_components=10, covariance_type='full', random_state=1)
gmm_model.fit(data)
print(gmm_model.converged_)


# Extract the means as well as the covariances
# Your code here
mns = gmm_model.means_
covs = gmm_model.covariances_


# Reshape the images
im = mns.reshape(10, 8, 8)

# Don't change this code
# Figure size in inches
fig = plt.figure(figsize=(8, 3))


# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    
    # Display images
    ax.imshow(im[i].reshape((8, 8)), cmap=cm.binary)
    plt.axis('off')

# Show the plot
plt.show()

print (mns.shape)
print (covs.shape)

# Your code here
samples = np.zeros((10,64))

for i in range(10):
#     samples[i] = covs[i,5]
    # samples[i] = np.random.multivariate_normal(mns[i], covs[i])
    samples[i] = .mv_normal.rvs(mns[i], covs[i])

print2(samples.shape)
im = samples.reshape(10,8,8)  


# Don't Change this code
# Figure size in inches
fig = plt.figure(figsize=(8, 3))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(im[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')

# Show the plot
plt.show()