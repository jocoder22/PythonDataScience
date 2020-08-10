#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# Import different modules for using with the notebook
from IPython.display import display
from IPython.display import Image
from IPython.display import HTML

from printdescribe import print2

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names
