# Importing Libraries 
import numpy as np 
import pandas as pd 
from scipy.linalg import lu, qr, cholesky
from numpy import linalg as LA

from printdescribe import print2

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = 10,8
plt.rcParams["axes.facecolor"] = "0.92"
np.random.seed(42)

# Create square matrix
A =  = np.arange(3,19, dtype=float).reshape(-1,4)
# Ai =  = np.arange(3,19).reshape(-1,4)
# perform eigendecomposition
val, vec = LA.eig(A)

# print the result
print2(val, vec)

# checking the result
# Au = lamda * u
Avec = A @ vec[:,0]
lambdavec = val[0] * vec[:,0]

# print the results
print2(Avec, lambdavec)


# Sorting the eigenvalues in ascending order
# first find the ascending indexes using argsort()
indx = np.argsort(val)

# for descending order use
# indx = np.argsort(val)[::-1]

# then apply the sorted indexes to eigenvalues and eiganvectors
val_sorted = val[indx]
vec_sorted = vec[:,indx]

# print the results
print2(val, vec, indx, val_sorted, vec_sorted)


# Recovering the original matrix
A_re = vec.dot(diag(val)).dot(inv(vec))
Q = vec
v_diagonal = diag(val)
Q_inverse = inv(vec)

# can sort the eiganvalues directly
val_sorted2 = np.sort(val)
print2(val_sorted2)

A_ = Q @ v_diagonal @ Q_inverse
print2(A, A_)

# check equality
print2(np.array_equal(A, A_))
print2(np.array_equiv(A, A_))
print2(np.allclose(A, A_))




import IPython.display
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from printdescribe import print2

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = 10,8
plt.rcParams["axes.facecolor"] = "0.92"

# tf.reset_default_graph()
reset = tf.reset_default_graph()

mat1 = tf.constant([[3., 4.]])
mat2 = tf.constant([[2.],[6.]])

print2(f"matrix 1: {mat1}", f"matrix 2: {mat2}")

product = tf.matmul(mat1, mat2)
print2(product)

reset
with tf.Session() as ses:
  result = ses.run(product)
  
 print2(result, result.shape, type(result))


reset
trend = tf.Variable(2, name="trender")
state = tf.Variable(3, name="counter")
update = tf.assign(state, state * trend)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print2(state.eval())
    
    for _ in range(3):
        sess.run(update)
        print(state.eval())
        
        val = state.eval()
        
print2(val)
