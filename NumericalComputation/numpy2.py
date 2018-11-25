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
I_array7 = np.identity(4) + 7
print(I_array, I_array.dtype)

# ones
ones = np.ones(5)
print(ones, ones.dtype)


# vector operations
myarray - zeroarray6
zeroarray6 / myarray
myarray - I_array7
I_array7 / myarray

# range, geomspace, logspace
Xarange = np.arange(4, 14, 2)
Xlinspace = np.linspace(2, 30, num=40)
print(Xlinspace)
Xlinspace2 = np.linspace(2, 30, num=29)
print(Xlinspace2)

Xgeomspace = np.geomspace(2, 5630, num=10)
print(Xgeomspace)

Xlogspace = np.logspace(2, 3, num=40)
print(Xlogspace)


np.logspace(3, 4, num=5)
np.logspace(np.log10(3), np.log10(4), num=5)


# logical operations
x = np.array([0, 1, 0, 0, 1], dtype=bool)
y = np.array([1, 1, 0, 1, 0], dtype=bool)
print(np.logical_or(x, y))

print(np.logical_and(x, y))

x = np.array([16, 34, 57, 17, 29])
print(np.logical_or(x < 18, x > 40))

longdis = np.arange(15).reshape(5, 3)


# some basic statistics
np.sum(longdis)
np.amin(longdis)
np.amax(longdis)
np.amin(longdis, axis=0)
np.amin(longdis, axis=1)
np.percentile(longdis, 75)
np.percentile(longdis, 95, axis=1)

np.argmax(longdis)
np.argmin(longdis)

# slicing
x = np.arange(14)
x[1:3]
x[1:3, ]
x[1:3:] # start from index 1 up to not including index 3, step 1
x[1:11:3] # start from index 1 up to not including index 11, step 3
x[:, 1:3]
x[1:3, 1:3]
x[[0, 1, 2], [0, 1, 3]]


# numpy: Resizing
x = np.arange(16).reshape(4, 4)  # only changes the output
np.resize(x, (2, 2))  # change the array permanently
np.resize(x, (6, 6))
x = np.reshape(np.arange(16), (4, 4))