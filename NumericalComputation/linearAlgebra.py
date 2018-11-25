import numpy as np
from numpy.linalg import multi_dot 

# Dot product
x1 = np.arange(20).reshape(5, 4)
x2 = np.arange(12).reshape(4, 3)
x3 = np.arange(18).reshape(3, 6)
x4 = np.arange(12).reshape(6, 2)

np.dot(x1, x2)
np.matmul(x1, x2)
x1@x2

multi_dot([x1, x2])
multi_dot([x1, x2, x3, x4])