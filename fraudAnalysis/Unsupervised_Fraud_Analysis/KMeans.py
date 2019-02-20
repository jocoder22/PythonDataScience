#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import seaborn as sns
plt.style.use('ggplot')



sp = '\n\n'

url2 = 'https://assets.datacamp.com/production/repositories/2162/datasets/08cfcd4158b3a758e72e9bd077a9e44fec9f773b/chapter_3.zip'

# download all the zip files
response = requests.get(url2)

# unzip the content
zipp = ZipFile(BytesIO(response.content))

# Dsiplay files names in the zip file
mylist = [filename for filename in zipp.namelist()]

# Load data to DataFrame from file_path:
data = pd.read_csv(zipp.open(mylist[5]))

labels = data['fraud']
data = data.drop(['Unnamed: 0',  'fraud'], axis=1)


# Turn data to float and scale
scaler = MinMaxScaler()

# data = np.array(data).astype(np.float)
data_scaled = scaler.fit_transform(data)


# find the number of clusters using the elbow method
n = range(1, 15)
kmm = [KMeans(n_clusters=i, random_state=12) for i in n]
kscores = [abs(kmm[k].fit(data_scaled).score(data_scaled)) for k in range(len(kmm))]

plt.plot(kscores, n)
plt.xlabel('K-scores (Absolute values)')
plt.ylabel('Number of cluster ')
plt.title('KMeans Elbow Curve')
plt.show()


# fit the KMeans models
km = MiniBatchKMeans(n_clusters=3, random_state=1).fit(data_scaled)

# find the prediction and the centroid location
cluster_labels = km.predict(data_scaled)
center = km.cluster_centers_

kvalues = pd.DataFrame(cluster_labels)
data.insert(0, "KMean_labels", kvalues)
print(kvalues.shape, data_scaled.shape, data.head(),sep=sp)

plt.scatter(range(0,data.shape[0]), data.amount, c=data.KMean_labels, s=20)
plt.show()


# setting outliers as fraud
distanceEul = [np.linalg.norm(x-y) for x,y in zip(data_scaled, center[cluster_labels])]

distanceEu = np.array(distanceEul)
distanceEu[distanceEul >= np.percentile(distanceEul, 95)] = 1
distanceEu[distanceEul < np.percentile(distanceEul, 99)] = 0


print(labels.shape, distanceEu.shape, sep=sp)
print(roc_auc_score(labels, distanceEu))


X_train, X_test, y_train, y_test = train_test_split(
    data_scaled, labels, test_size=0.3, random_state=0)

# Define K-means model
kmeans = MiniBatchKMeans(n_clusters=3, random_state=22).fit(X_train)

# Obtain predictions and calculate distance from cluster centroid
X_test_clusters = kmeans.predict(X_test)
X_test_clusters_centers = kmeans.cluster_centers_
distance2 = [np.linalg.norm(x-y) for x, y in zip(X_test,
                                            X_test_clusters_centers[X_test_clusters])]

# Create fraud predictions based on outliers on clusters
kmpred = np.array(distance2)
kmpred[distance2 >= np.percentile(distance2, 95)] = 1
kmpred[distance2 < np.percentile(distance2, 95)] = 0
print(roc_auc_score(y_test, kmpred))


# Create a confusion matrix
km_cm = confusion_matrix(y_test, kmpred)
print(km_cm)


def plot_confusion_matrix(km_cm):
  df_cm = pd.DataFrame(km_cm, ['True Normal', 'True Fraud'], [
                       'Pred Normal', 'Pred Fraud'])
  plt.figure(figsize=(8, 4))
  sns.set(font_scale=1.4)
  sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')
  plt.show()

plot_confusion_matrix(km_cm)


