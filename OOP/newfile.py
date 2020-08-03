import numpy as np

A = np.array([[2,1],[4,1],[1,1]])
b = np.array([[9],[17],[5]])
m. p = np.linalg.inv(A.T @ A) @ A.T @ b
