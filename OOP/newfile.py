import numpy as np
np.random.seed(901)

A = np.array([[2,1],[4,1],[1,1]])
b = np.array([[9],[17],[5]])
m. p = np.linalg.inv(A.T @ A) @ A.T @ b


X = np.array([2,4,1])
y = np.array([9,17,5])
m = (np.mean(X) * np.mean(y) - np.mean(X*y)) / (np.mean(X)**2 - np.mean(X**2))
b = np.mean(y) - m * np.mean(X)

print2(f"Value of m: {m}", f"Value of b: {b}")



x_train = np.linspace(0,1,100)
y_train = 0.2*x_train + 1 + 0.01*np.random.randn(x_train.shape[0])

X = np.array(x_train)
y = np.array(y_train)

b = np.asarray([1]) # hint: np.asarray()
col2 = np.ones((X.shape)) # hint: np.ones()
A = np.vstack((X, col2)).T # hint: np.vstack().T

# Normal equations
A_normal = np.linalg.inv(A.T @ A)
b_normal = A.T @ y

# Solve
m, b = A_normal @ b_normal

# The computed values of m and b should be compared with the values
# m = 0.2 and b = 1.0, used to generate the data
print2(f"Value of m: {m}", f"Value of b: {b}")
