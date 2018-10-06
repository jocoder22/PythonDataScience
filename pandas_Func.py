import pandas as pd 
from pandas import Series, DataFrame
import numpy as np 

fser100 = Series(np.arange(6))
fser200 = Series([90, 34, 67, 12, 100, 79])
fser300 = Series([50, 24, 56, 89, 33],
                 index=[0, 1, 2, 3, 5])