from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
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
pca2.components

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

pca2e = pca2.explained_variance_ratio_.sum()
pca3e = pca3.explained_variance_ratio_.sum()

# Percentage explained
print("PCA 2-components explains {:.2f}%".format(pca2e*100))
print("PCA 3-components explains {:.2f}%".format(pca3e*100))


# find components making up at least 95%
pca95 = PCA(n_components=0.95)
pca95pc = pca95.fit_transform(iris.data)
print(pca95.explained_variance_ratio_.sum())
print(pca95pc.shape)


# Apply PCA on Boston Housing data
boston = load_boston()
print(boston.data.shape)

pca95b = PCA(n_components=0.95)
pca95pcb = pca95b.fit_transform(boston.data)
print(pca95b.explained_variance_ratio_.sum())
print(pca95pcb.shape)


# Randomized PCA
rpca2 = PCA(svd_solver='randomized', n_components=2)
rpca2c = rpca2.fit_transform(boston.data)
plt.scatter(rpca2c[:, 0], rpca2c[:, 1], c=boston.target, 
            alpha=0.8, s=60, marker='o', edgecolors='white')
plt.show()
rpca2.explained_variance_ratio_.sum()

