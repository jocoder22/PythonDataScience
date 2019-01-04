#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

print(os.getcwd())
os.chdir('c:\\Users\\Jose\\Desktop\\')

schema = np.dtype([('Name', '<U16'),
                   ('Age',  np.float16),
                   ('Gender', '<U16'),
                   ('City', '<U16')])
data = np.loadtxt("Desktop\\test1.txt",
                  skiprows=1, dtype=schema, delimiter=',')

data =  np.loadtxt("text1.txt",skiprows=1, dtype=schema, delimiter=',')
data  # this is a one dimensional numpy array
data.shape # ==> (3,)
type(data)

data_multiD = np.reshape(data, (3, -1))
data_multiD.shape  # (3, 1)


# Slicing;
data[:5]
data[:5]['Name']
data[:5][['Name', 'Age']]


# Using pandas
data2 = pd.read_csv("people.csv")
data2
type(data2)  # <class 'pandas.core.frame.DataFrame'>
data2.head()

# pandas slicing;
data2.head().name
data2.head().age
data2.Name

data2.index = data2['name']

data2.head().loc[:, ['height', 'age', 'name']]
type(data2.Name)  # <class 'pandas.core.series.Series'>


# pandas built on numpy;
dav = data2.values
type(dav)  # <class 'numpy.ndarray'>


# Form pandas dataframe from numpy array;
newdata = pd.DataFrame(data)
newdata
type(newdata)  # <class 'pandas.core.frame.DataFrame'>
