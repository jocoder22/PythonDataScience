import numpy as np

# Create numpy array
list1 = [3, 6, 9, 12]
myarray = np.array(list1)
print(type(myarray))

# Scalar arthmetics
print(myarray + 2)
print(myarray - 1)
print(myarray * 4)
print(myarray / 2)

# check dtype
print((myarray + 2).dtype)
print((myarray - 1).dtype)
print((myarray * 4).dtype)
print((myarray / 3).dtype)
