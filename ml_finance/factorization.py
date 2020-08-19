
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
print2(P,L,U)
