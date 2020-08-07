#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import seaborn as sns
from printdescribe import print2


genes = ["gene"+str(i) for i in range(1,101)]
wt = ["wt"+str(i) for i in range(1,7)]
ko = ["ko"+str(i) for i in range(1,7)]
data = pd.DataFrame(columns=[*wt ,*ko], index=genes)

print2( wt, ko)
data = pd.DataFrame(columns=[*wt ,*ko], index=genes)
print2(data.head())

n = 2
for gene in data.index:
    data.loc[gene, :"wt6"] =  np.random.poisson(
        lam=np.random.randint(10,100),size=6)
    np.random.seed(90 + n)
    data.loc[gene, "ko1":] =  np.random.poisson(
        lam=np.random.randint(10,100),size=6)
    n += 5
    

scaled_data = preprocessing.scale(data.T)
scaled_data[:20]

  
   
scaler =  StandardScaler()
dataset = scaler.fit_transform(data.T)
print(dataset[:20])


pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_*100, 1)
labels = ["PC"+str(i) for i in range(1,len(per_var)+1)]
plt.bar(labels,per_var)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Pricipal Component")
plt.title("Scree Plot")
plt.show()


pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], 
                      columns=labels)
plt.scatter(pca_df.PC1, pca_df.PC2)


for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], 
                          pca_df.PC2.loc[sample]))
    
    
plt.ylabel(f"PC2 - {per_var[1]}")
plt.xlabel(f"PC1 - {per_var[0]}")
plt.title("PCA Plot")
plt.show()


loading_scores = pd.Series(pca.components_[0], index=genes)
sorted_scores = loading_scores.abs().sort_values(ascending=False)
top_ten = sorted_scores[:10].index.values
print2(loading_scores[top_ten], 
       loading_scores[top_ten].sort_values(ascending=False))
