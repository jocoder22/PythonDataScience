#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
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


# Initialize and fit the DBscan model
db = DBSCAN(eps=0.9, min_samples=10, n_jobs=-1).fit(data_scaled)

# Obtain the predicted labels and calculate number of clusters
pred_labels = db.labels_
n_clusters = len(set(pred_labels)) - (1 if -1 in labels else 0)

# Print performance metrics for DBscan
print('Estimated number of clusters: %d' % n_clusters)
print("Homogeneity: %0.3f" % homogeneity_score(labels, pred_labels))
print("Silhouette Coefficient: %0.3f" %
      silhouette_score(data_scaled, pred_labels))


# Count observations in each cluster number
counts = np.bincount(pred_labels[pred_labels >= 0])

# Print the result
print(counts)


# Count observations in each cluster number
counts = np.bincount(pred_labels[pred_labels >= 0])

# Sort the sample counts of the clusters and take the top 3 smallest clusters
smallest_clusters = np.argsort(counts)[:3]

# Print the results
print("The smallest clusters are clusters:")
print(smallest_clusters)

# Print the counts of the smallest clusters only
print("Their counts are:")
print(counts[smallest_clusters])

kk = dict()

for i in pred_labels:
    if i in kk.keys():
        kk[i] += 1
    else:
        kk[i] = 1


sortedKK = sorted(kk.items(), key=lambda k: k[1])
print(kk, sortedKK, sep=sp)


# Create a dataframe of the predicted cluster numbers and fraud labels
df = pd.DataFrame({'clusternr': pred_labels, 'fraud': labels})

# Create a condition flagging fraud for the smallest clusters
df['predicted_fraud'] = np.where((df['clusternr'] == 9) | 
    (df['clusternr'] == 17) | (df['clusternr'] == 21) |
     (df['clusternr'] == 11) | (df['clusternr'] == 20) |
    (df['clusternr'] == 19) | (df['clusternr'] == 13) |
    (df['clusternr'] == 18), 1, 0)

# Run a crosstab on the results
print(pd.crosstab(df.fraud, df['predicted_fraud'], rownames=[
      'Actual Fraud'], colnames=['Flagged Fraud']))
