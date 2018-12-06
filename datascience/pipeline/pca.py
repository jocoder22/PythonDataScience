from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
covdata = np.corrcoef(iris.data.T)
print(iris.feature_names)
print(covdata)


# pca