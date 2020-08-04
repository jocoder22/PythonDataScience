import numpy as np
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from printdescribe import print2

np.random.seed(901)

A = np.array([[2,1],[4,1],[1,1]])
b = np.array([[9],[17],[5]])
m. p = np.linalg.inv(A.T @ A) @ A.T @ b


X = np.array([2,4,1])
y = np.array([9,17,5])
m = (np.mean(X) * np.mean(y) - np.mean(X*y)) / (np.mean(X)**2 - np.mean(X**2))
b = np.mean(y) - m * np.mean(X)

print2(f"Value of m: {m}", f"Value of b: {b}")


# create data for training
X = np.linspace(0,1,100)
y = 0.2*X + 1 + 0.01*np.random.randn(X.shape[0])


# prepare data for normal equation
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


# QR factorization using numpy
# Your code here 
Q, R = np.linalg.qr(A)
# m, b = np.linalg.inv(R) @ Q.T @ y
m, b = np.linalg.inv(R).dot(Q.T.dot(y))


print2(Q.shape, f'R = {R}', f'Q^TQ = {np.dot(Q.T,Q)}')
print2(f"Value of m: {m}", f"Value of b: {b}")


# finding the beta using Scikit-learn
reg = LinearRegression().fit(X.reshape(-1, 1), y)
reg.score(X, y)
m, b = reg.coef_[0],reg.intercept_

print2(f"Value of m: {m}", f"Value of b: {b}")


# finding the coefficients using Keras
model = Sequential()
model.add(Dense(1, input_dim=1, kernel_initializer='normal', activation='linear'))

# Compile model
model.compile(loss="mean squared error", optimizer=Adam(lr=0.01), metrics=['mse'])

# Fit model: use a batch_size=20, epochs=300
model.fit(x=x_train, y=y_train, batch_size=, epochs=, verbose=1)
