#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import requests

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import seaborn as sns
from printdescribe import print2

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'


colname = '''MPG Cylinders Displacement Horsepower Weight Acceleration 
             Model_year Origin Car_name'''.split()

dataset = pd.read_csv(url, names=colname, na_values=['na','Na', '?'],
                skipinitialspace=True, comment='\t', sep=" ", quotechar='"')

print2(dataset.head, dataset.shape)
describe2(dataset.iloc[:,:6])

dataset.drop(columns='Car_name', inplace=True)
print2(dataset.isna().sum())
dataset.dropna(inplace=True) 
