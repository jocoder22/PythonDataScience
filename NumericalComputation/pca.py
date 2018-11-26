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