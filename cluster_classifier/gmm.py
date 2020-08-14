#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM

from printdescribe import print2

# download the data
iris = datasets.load_iris()

data = iris.data
labels = iris.target
labelnames = iris.target_names
