from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
covdata = np.corrcoef(iris.data.T)
print(iris.feature_names)
print(covdata)


# pca
# ## 2 components
pca2 = PCA(n_components=2)
pca2comp = pca2.fit_transform(iris.data)
pca2comp.shape

plt.scatter(pca2comp[:, 0], pca2comp[:, 1], c=iris.target, alpha=0.8, 
            s=60, marker='o', edgecolors='white')
plt.show()


# ## 3 components
pca3 = PCA(n_components=3)
pca3comp = pca3.fit_transform(iris.data)
pca3comp.shape

plt.scatter(pca3comp[:, 0], pca3comp[:, 1], c=iris.target, alpha=0.8, 
            s=60, marker='o', edgecolors='white')
plt.show()

pca2 = pca2comp.explained_variance_ratio_.sum()
pca3 = pca3comp.explained_variance_ratio_.sum()

