import numpy as np
import pandas as pd

schema = np.dtype([('Name', '<U16'),
                   ('Age',  np.float16),
                   ('Gender', '<U16'),
                   ('City', '<U16')])
data4 = np.loadtxt("/Documents/test1.txt",
                   skiprows=1, dtype=schema, delimiter=',')

