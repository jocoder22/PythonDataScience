#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn.cluster import KMeans

from printdescribe import print2

nn = np.array([-8.0,-3.0, 0.0, 6.5, 9.0, 45.5]).reshape(-1,1)
cc = np.array([-1.5,-1.0,-0.5,1.5,2.0,2.5]).reshape(-1,1)
gm = GaussianMixture(n_components=2, covariance_type='full')
gmm = gm.fit(nn)


pred = gmm.fit_predict(nn)
logprob = gmm.score_samples(nn)
responsibilities = gmm.predict_proba(nn)
pdf = np.exp(logprob)
pdf_individual = responsibilities * pdf[:, np.newaxis]

print2(nn, logprob, responsibilities, pdf, pdf_individual)
print2(np.round(gmm.weights_, 2), np.round(gmm.means_, 2), 
        np.round(gmm.covariances_, 2))

print2(np.round(gmm.precisions_,2), np.round(gmm.precisions_cholesky_, 2))
print2(np.round(pred, 2))
print2(gmm.score(nn))


print("######################################################")
print("This is for  [-1.5,-1.0,-0.5,1.5,2.0,2.5] data")
gcc = gm.fit(cc)
pred = gcc.fit_predict(cc)
logprob = gcc.score_samples(cc)
responsibilities = gcc.predict_proba(cc)
pdf = np.exp(logprob)
pdf_individual = responsibilities * pdf[:, np.newaxis]

print2(f"The data : {cc}", logprob, 
    f"The responsibility: {responsibilities}", pdf, pdf_individual)
print2(np.round(gcc.weights_, 2), np.round(gcc.means_, 2), 
        np.round(gcc.covariances_, 2))

print2(np.round(gcc.precisions_,2), np.round(gcc.precisions_cholesky_, 2))
print2(np.round(pred, 2))
print2(gcc.score(nn))



###########################################################################
############# kmeans ######################################################
cc = np.array([-1.5,-1.0,-0.5,1.5,2.0,2.5]).reshape(-1,1)
bb = np.array([-8.0,-3.0,0.0,6.5,9.0,45.5]).reshape(-1,1)

kmean = KMeans(n_clusters=2, random_state=42).fit(bb)
center, labels = kmean.cluster_centers_, kmean.labels_
print2(center, labels)


kmean = KMeans(n_clusters=2, random_state=42).fit(cc)
center, labels = kmean.cluster_centers_, kmean.labels_
print2(center, labels)

print(cc[3:].var())


pth = "https://www.analyticsvidhya.com/wp-content/uploads/2019/10/Clustering_gmm.csv"
pf = pd.read_csv(pth)
print2(pf.head, pf.shape)