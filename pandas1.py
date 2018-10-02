import numpy as np
import pandas as pd

schema = np.dtype([('Name', '<U16'),
                   ('Age',  np.float16),
                   ('Gender', '<U16'),
                   ('City', '<U16')])
data = np.loadtxt("/Documents/test1.txt",
                  skiprows=1, dtype=schema, delimiter=',')

data  # this is a one dimensional numpy array
data.shape
type(data)


# Slicing;
data[:5]
data[:5]['Name']
data[:5][['Name', 'Age']]


# Using pandas;
data2 = pd.read_csv("/Documents/test1.txt")
data2
type(data2)  # <class 'pandas.core.frame.DataFrame'>
data2.head()

# pandas slicing;
data2.head().Name
data2.head().Gender
data2.Name

data2.head().loc[:, ['Gender', 'Age', 'Name']]
type(data2.Name)  # <class 'pandas.core.series.Series'>