import numpy as np

# Create numpy array
list1 = [3, 6, 9, 12]
myarray = np.array(list1)
print(type(myarray))

# Scalar arithmetics
print(myarray + 2)
print(myarray - 1)
print(myarray * 4)
print(myarray / 2)

# check dtype
print((myarray + 2).dtype)
print((myarray - 1).dtype)
print((myarray * 4).dtype)
print((myarray / 3).dtype)

# zero array
zeroarray = np.zeros(4)
zeroarray6 = np.zeros(4) + 6
print(zeroarray, zeroarray6)
print(zeroarray.dtype, zeroarray6.dtype)

# Indentity
I_array = np.identity(4)
print(I_array, I_array.dtype)

# ones
ones = np.ones(5)
print(ones, ones.dtype)

