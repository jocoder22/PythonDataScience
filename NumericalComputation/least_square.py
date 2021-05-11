import numpy as np
from sympy import *
import pandas as pd
from scipy.linalg import lu
from IPython.display import display, Latex

# create arrays of independent variables
bb = np.array([[1,5],[2,3],[3,6]])

# create array of 1, the intercepts
icp = np.ones(3).reshape(-1,1)

# append the independent variables to the intercept
tt = np.append(icp, bb, axis = 1)

# the normal equation is A.T * A * x = A.T * b
ttTtt = tt.T@tt
ttTb = tt.T@b

# append the results
ttaa = np.append(ttTtt, ttTb.reshape(-1,1), axis=1)

# find the LU decomposition
p, l, u = lu(ttaa)

# use the Lowe triangle to extract the beta-coefficients
result = []
for i in range(len(l)):
    result.append(l[i][0])

# create
parameter_symbols = [rf'$\beta_{b}$' for b in range(len(result))]

df = pd.DataFrame([result],index = ["Results"], columns=parameter_symbols)

print(df)
