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