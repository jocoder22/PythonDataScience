import numpy as np


a = np.array([[3, 6, 8], [0, 1, 5], [2, 9, 8]])
b = np.array([3, 8, 6])

# Using the inverse method
a_inverse = np.linalg.inv(a)
solution = np.dot(a_inverse, b)

# Using the solve function
solution2 = np.linalg.solve(a, b)

# check for equality
np.allclose(solution, solution2)  # True


# Grandient
ga = np.array([5, 8, 12, 4, 9])
gb = np.array([5, 8, 3, 12, 4, 1,  9, 8, 20, 0, 10, 6]).reshape(3, 4)

gag = np.gradient(ga)
gbg = np.gradient(gb)