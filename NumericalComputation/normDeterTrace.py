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
# L1- norm is for the calculation of mean-absolute error (MAE) 
# formula: MAE(x1, x2) = 1/n(||x1 - x2||)
np.linalg.norm(L01, ord=1)  # 10
np.linalg.norm(L02, ord=1)  # 21
np.linalg.norm(L03, ord=1)  # 31


# norm 2, L2: the Euclidean norm
# this Calulates the length of the vector
# L2-norm is used in the mean-squared error (MSE) calculations.
np.linalg.norm(L01, ord=2)  # 5.4772255750516612
np.linalg.norm(L02, ord=2)  # 11.704699910719626
np.linalg.norm(L03, ord=2)  # 13.820274961085254


# norms for Matrix
# The norm of the matrix reveals how much that 
# matrix could possibly stretch a vector
# norms are very important for distance metrics  k-nearest 
# neighbors algorithm (k-NN) and in k-means clustering as well.
mat1 = np.array([12, 3, 4, 6, 7, 34, 21, 9, 4, 2, 9, 7]).reshape(4, 3)


#  First order norm for matrix
#  calculated the max element as column-wise first and
#  gives the maximum result in all columns, which will 
#  be the norm of the matrix for the first order. 
np.linalg.norm(mat1, ord=1)  # 49


#  Infinity order norm for matrix
#  calculated the max element as row-wise first and 
#  gives the maximum result in all rowe, which will 
# be the norm of the matrix for the infinity order. 
np.linalg.norm(mat1, np.inf)  # 47


#  Euclidean/Frobenius norm for matrix
np.linalg.norm(mat1, ord=2)  # 39.254426189588429


# Determinant
# the determinant is the scaling factor of
# a matrix in linear transformation.
mat2 = np.array([12, 3, 4, 6, 7, 34, 21, 9, 4, 2, 9, 7]).reshape(3, 2, 2)
mat3 = np.array([3, 4, 6, 8, 5, 13, 23, 9, 10]).reshape(3, 3)
np.linalg.det(mat2)
np.linalg.det(mat3)


# Trace of a matrix
# the trace is the sum of diagonal elements of a matrix
np.trace(mat1)  # 23
np.trace(mat2)  # array([33, 12])
np.trace(mat3)  # 18