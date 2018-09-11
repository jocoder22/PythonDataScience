import numpy as np

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


# create character array
char1 = np.array([['Cup', 'Spoon', 'knives'], ['Table', 'Chairs', 'Beds']])
char1.dtype  # dtype('<U6')
print(char1)
