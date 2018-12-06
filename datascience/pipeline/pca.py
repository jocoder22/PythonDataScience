from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
covdata = np.corrcoef(iris.data.T)
print(iris.feature_names)
print(covdata)


# pca
pca2 = PCA(n_components=2)
pca2comp = pca2.fit_transform(iris.data)
pca2comp.shape 