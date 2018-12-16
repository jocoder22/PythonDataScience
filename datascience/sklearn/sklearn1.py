import numpy as np 
from sklearn import datasets

irisdata = datasets.load_iris()
features_ = irisdata.data[:, [2, 3]]
target_ = irisdata.target


