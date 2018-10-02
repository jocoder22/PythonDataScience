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
