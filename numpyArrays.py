import numpy as np
# Find current working directory
import os
print(os.getcwd())

# Create arrays of  ones, all integers

Vones = np.ones((4), dtype=np.int8)
Vones2 = np.ones((4, 6), dtype=np.int8)


# converting dtype
Vones3 = np.ones((9))
Vones3.dtype  # dtype('float64')
Vones3 = np.array(Vones3, dtype=np.int8)


# Create matrix of random integers
mat1 = np.random.randn(5, 5)
mat2 = np.random.randn(2, 3, 2)
print(mat2.shape)  # show the dimensions of the array


# create character array
char1 = np.array([['Cup', 'Spoon', 'knives'], ['Table', 'Chairs', 'Beds']])
char1.dtype  # dtype('<U6')
print(char1)

char2 = np.array(['IP10mg', 'Placebo', 'IP10mg', 'Placebo', 'IP20mg'],
                 dtype='<U12')
char2.type
print(char2)
print(char2.shape)


# Using tuples
tup1 = np.array([[(2, 3, 5), (9, 'man', 8)], [(9, 9, 4), ('Ken', 5, 3)]])
tup1.dtype
print(tup1)

# coping np array
tup2 = tup1  # this actually copy, but establish link to tup1
tup1[0, 0] = 'Josp'  # this will update values in tup2 also

# To actually copy, using copy() method
tup3 = tup1.copy()
tup1[0, 0] = 'Josphine'  # this wouldn't update values in tup3

# find current working directory
os.getcwd()


# saving array, saved into .npy
np.save('Tup1', tup1)
np.savetxt('mat1.txt', mat1, delimiter=',')

# print contents of saved files
savmat1 = open('mat1.txt', 'r')
for data in savmat1:
    print(data)

savmat1.close()


# loading data into current session
tup4 = np.load('Tup1.npy')
print(tup4)


matt1 = np.loadtxt('mat1.txt', delimiter=',')
print(matt1)


# Slicing numpy array
# Very important to remember the dimensions of the array
# select all rows in mat1 = np.random.randn(5, 5)
mat1[3, 2]
matrow = mat1[0:4, :]   # or matrow = mat1[:4, :] or matrow = mat[:-1, :]
                        # this is up to but not including the 4th row


# The 3rd option to indicate increments
matrowin = mat1[:4:2, :]  # select every 2nd row


# Don't flatten, retain the shape of the array in one column array
# use list to keep the column
matRshape = mat1[:4, [1]]

# Reversing matrix
# Reverse rows
mat1[::-1, :]  # here last row will be first while the last row becomes the first

# Reverse columns
mat1[:, ::-1]   # last column now first

# Reverse rows and columns
mat1[::-1, ::-1]


# Retaining original dimension
# Use np.newaxis to add new extra dimension
mat3d = np.random.randn(3, 4, 2)
mat3d1 = mat3d[:, 1, :]  # this is two dimensional
print(mat3d1.shape)   # return (2, 3) => two dimensional

mat3dd = mat3d[:, 1, np.newaxis, :]
print(mat3dd.shape)  # return (2, 1, 3) => three dimensional
