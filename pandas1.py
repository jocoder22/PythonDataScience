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
data = np.loadtxt("/Documents/test1.txt",
                  skiprows=1, dtype=schema, delimiter=',')

data =  np.loadtxt("Desktop/people.csv", delimiter=',')
data  # this is a one dimensional numpy array
data.shape
type(data)


# Slicing;
data[:5]
data[:5]['Name']
data[:5][['Name', 'Age']]


# Using pandas
data2 = pd.read_csv("/Documents/test1.txt")
data2 = pd.read_csv("people.csv")
data2
type(data2)  # <class 'pandas.core.frame.DataFrame'>
data2.head()

# pandas slicing;
data2.head().Name
data2.head().Gender
data2.Name

data2.index = data2['Name']

data2.head().loc[:, ['Gender', 'Age', 'Name']]
type(data2.Name)  # <class 'pandas.core.series.Series'>


# pandas built on numpy;
dav = data2.values
type(dav)  # <class 'numpy.ndarray'>


# Form pandas dataframe from numpy array;
newdata = pd.DataFrame(data)
newdata
type(newdata)  # <class 'pandas.core.frame.DataFrame'>
