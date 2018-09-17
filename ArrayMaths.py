import numpy as np
from numpy.random import randn
import numpy.linalg as ln


radN1 = np.array(randn(2, 3, 4) * 8, dtype=np.int64)
radN2 = np.array(randn(3, 4, 6) * 9, dtype=np.int64)
arr1 = np.numpy([[3, 5, 7], [7, 9, 12], [5, 2, 5]])
arr2 = ln(90)


# Array maths
arr3 = radN2 + 10
arr4 = radN1 / 6
arr5 = radN2 ** 2
arr6 = radN1 * radN2
arr7 = radN1 ** (radN2 / 3)


# Array Broadcasting
radN3 = np.array(randn(2, 3, 4) * 4, dtype=np.int64)
radN4 = np.array([[[5, 8, 9, 1]]])
print(radN3.shape)  # (2, 3, 4)
print(radN4.shape)  # (1, 1, 4)
arr7 = radN3 * radN4


radN5 = np.array([[[6], [1], [4]]])
radN6 = np.array([[[6]], [[1]], [[4]]])
radN5.shape  # (1, 3, 1)
arr8 = radN3 * radN5


radN2.shape  # (3, 4, 6)
radN6.shape  # (3, 1, 1)
print(radN2 * radN6)
