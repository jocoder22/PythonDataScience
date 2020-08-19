
# Importing Libraries 
import numpy as np 
import pandas as pd 
from printdescribe import print2

import pandas_datareader.data as dr
import matplotlib
import matplotlib.pyplot as plt 
import math
import statsmodels.tsa.stattools as ts
import numpy as np 

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



