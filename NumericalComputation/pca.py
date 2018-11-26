import numpy as np 
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler 

datasets = datasets.load_breast_cancer() 
cancerdataset = datasets.data
cancerdataset = StandardScaler().fit_transform(cancerdataset)
cancerdataset.shape


coval_matrix = np.cov(cancerdataset, rowvar=False)
coval_matrix.shape

eig_value, eig_vector = np.linalg.eig(coval_matrix)
eig_pairs = [(np.abs(eig_value[i]), eig_vector[:,i]) for i in
             range(len(eig_value))]

Sorted_pairs = eig_pairs.sort(key=lambda x: x[0], reverse=True)
for i in eig_pairs:
    print(i[0])


matrix_w = np.hstack((eig_pairs[0][1].reshape(30,1), eig_pairs[1][1].reshape(30,1)))
matrix_w.shape 
transformed = matrix_w.T.dot(cancerdataset.T) 
transformed = transformed.T 
transformed[0]
transformed.shape


# Using built-in fuctions
pca2 = decomposition.PCA(n_components=2)
stand_data = StandardScaler().fit_transform(datasets.data)
transformed_pca = pca2.fit_transform(stand_data)
transformed_pca[0]