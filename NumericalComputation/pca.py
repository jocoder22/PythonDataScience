import numpy as np 
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler 

datasets = datasets.load_breast_cancer() 
cancerdataset = datasets.data
cancerdataset = StandardScaler().fit_transform(cancerdataset)
