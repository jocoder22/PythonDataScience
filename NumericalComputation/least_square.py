import numpy as np
from sympy import *
import pandas as pd


bb = np.array([[1,5],[2,3],[3,6]])
tt = np.append(np.ones(3).reshape(-1,1),bb, axis = 1)

ttTtt = tt.T@tt
ttTb = tt.T@b
