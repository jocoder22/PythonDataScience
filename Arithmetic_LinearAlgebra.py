import numpy as np
from numpy.random import randn

myarr1 = np.array(randn(5, 5) * 20, dtype=np.int8)
print(myarr1)

myarr1.tolist()  # turn the arrray into a list
myarr1.flatten()  # turn array into a single list


# Create empty array and file it with value;
EmptyArray = np.empty((3, 4), dtype=np.dtype('<U16'))

# fill array with value;
EmptyArray.fill('Goodies')


# sum array;
myarr1.sum()  # sum all the values
myarr1.sum(axis=0)  # sum along the rows
myarr1.sum(axis=1)  # sum along the columns


# Cumsum function;
myarr1.fill(100)  # fill myarr1 with 100

myarr1.cumsum(axis=0)  # do running sum along the rows
myarr1.cumsum(axis=1)  # do running sum along the columns


# Add mean function;
myarr2 = np.array(randn(5, 4) * 40, dtype=np.int8)
myarr2.mean(axis=0)
myarr2.mean(axis=1)

# using ufuncs;
print(myarr2)
""" array([[-40,   3, -45,   7],
            [  3, -16, -74, -51],
            [ 19,  24, -22, -21],
            [-14, -14, -61, -16],
            [-41,   0,  91, -32]], dtype=int8)
"""

np.sign(myarr2)  # find the sign of elements of myarr2 i.e -1, 0, 1
"""array([[-1,  1, -1,  1],
            [ 1, -1, -1, -1],
            [ 1,  1, -1, -1],
            [-1, -1, -1, -1],
            [-1,  0,  1, -1]], dtype=int8)
"""

