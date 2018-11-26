import numpy as np 
# norm 0, L0: the cardinality of a vector
# Total number of non-zero elements
L01 = np.arange(5)
L02 = np.array([2, 4, 6, 0, 9, 0])
L03 = np.array([3, 5, 6, 2, 0, 6, 0, 9, 0])

np.linalg.norm(L01, ord=0)  # 4
np.linalg.norm(L02, ord=0)  # 4
np.linalg.norm(L03, ord=0)  # 6


# norm 1, L1: the Taxicab norm or Manhattan norm
# Calulates the rectilinear distances of the vectors
np.linalg.norm(L01, ord=1)  # 10
np.linalg.norm(L02, ord=1)  # 21
np.linalg.norm(L03, ord=1)  # 31