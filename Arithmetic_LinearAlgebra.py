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

myarr2 * np.sign(myarr2)  # this creates absolute values of the array


# using vectorize() function;


def add20(num):
    if num > 0:
        return num + 20
    else:
        return 0


add20(14)  # 34
add20(-12)  # 0
add20(myarr2)  # Error

# using the vectorize function;
add20vect = np.vectorize(add20)
add20vect(myarr2)


# using faster and efficient numpy function;


def add20vem(arr):
    return_arr = arr.copy()
    return_arr[arr <= 0] = 0
    return return_arr


add20vem(myarr2)


# Timing the two processes;
import timeit

nnn = """
from numpy.random import randn
import numpy as np
"""

codd = """
my = np.array(randn(5, 4) * 40, dtype=np.int8)

def add20vem(arr):
    return_arr = arr.copy()
    return_arr[arr <= 0] = 0
    return return_arr

add20vem(my)
"""

cott = """
my = np.array(randn(5, 4) * 40, dtype=np.int8)
def add20(num):
    if num > 0:
        return num + 20
    else:
        return 0

add20vect = np.vectorize(add20)
add20vect(my)
"""

timeit.timeit(setup=nnn,
              stmt=codd,
              number=1000000)  # 8.151

timeit.timeit(setup=nnn,
              stmt=cott,
              number=1000000)  # 40.56