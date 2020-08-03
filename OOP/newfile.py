import numpy as np

A = np.array([[2,1],[4,1],[1,1]])
b = np.array([[9],[17],[5]])
m. p = np.linalg.inv(A.T @ A) @ A.T @ b


X = np.array([2,4,1])
y = np.array([9,17,5])
m = (np.mean(X) * np.mean(y) - np.mean(X*y)) / (np.mean(X)**2 - np.mean(X**2))
b = np.mean(y) - m * np.mean(X)

print(m,b)


