import pandas as pd 
from pandas import Series, DataFrame 
import numpy as np 

# create DataFrame;
data1 = DataFrame(np.arange(24).reshape(8, 3),
                  columns=["Age", "Height", "Weight"])

data1 - data1.loc[:,["Age","Weight"]]
data1.mean()
data1.std()

# for standardization;
(data1 - data1.mean())/data1.std()
data1.sum()


# Vectorization;
myobj = {'Baby': [10, 90, 40],
         'Girl': [22, 32, 25],
         'Boy': [54, 56, 67], 
         'Woman': ['Pen', 'Book', 'kett'],
         'Man': [89, 90, 78]}

data2 = DataFrame(myobj)

np.sqrt(data1)

# Mixed data type problem;
np.sqrt(data2)  # error: all column must be numeric
# use the select_dtypes() method to select only numeric;
np.sqrt(data2.select_dtypes([np.number]))

@np.vectorize
def trunc(x):
    return x if x > 12 else 0

@np.vectorize
def trunc2(x):
    return x if x > 30 else 0

trunc(data1)
# Mixed data type problem
trunc2(data2)  # Error, must all be of same type
trunc2(data2.select_dtypes([np.number]))

