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

# create matrix
A = A = np.linspace(1,9,9).reshape(-1,3)
B = np.arange(10,26).reshape(-1,4)

# PLU decomposition; used for square matrix
P,L, U = lu(A)
Pb,Lb, Ub = lu(B)

# print results
print2(P,L,U)
print2(Pb,Lb,Ub)

# recombine the triangular factor matrices
A_ = P @ L @ U
B_ = Pb @ Lb @ Ub

# print results
print2(A_, B_)


# QR decomposition for all matrix
# create matrix
A2 = np.linspace(1,35,35).reshape(-1,5)
B2 = np.arange(10,34).reshape(-1,4)
print2(A2, B2)

# perfom QR decomposition
Q,R = qr(A2)
Qb,Rb = qr(B2)

# print results
print2(Q, R)
print2(Qb, Rb)

# recombine factors
A2_ = Q @ R
B2_ = Qb @ Rb

# print results
print2(A2_, B2_)


#####################################################################################
#####################################################################################
# cholesky decomposition, for
# 1. square symmetric matrix
# 2. All eigenvalues are greater than zero
# ==> positive definite matrices
# create positive definite square symmetric matrices
#####################################################################################
#####################################################################################

n = 5
matrix1 = np.random.randint(20,30,size=(N,N))
A = (b + b.T)/2
sa = A @ A.T


# check for definite positivity
# w = list of eigenvalues
# v = colums of eigenvectors, one column per eigenvalues
w,v = LA.eig(sa)
print2(v)


# perform cholesky decomposition
# numpy.linalg.cholesky gives lower triangle
chola_numpy = LA.cholesky(sa)
chola_numpy_ = chola_numpy @ chola_numpy.T
print2(chola_numpy, chola_numpy_, sa)


# while scipy.linalg.cholesky gives upper triangle
chola = cholesky(sa)
chola_ = chola.T @ chola
print2(chola, chola_, sa)
