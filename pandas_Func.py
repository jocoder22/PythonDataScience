import pandas as pd 
from pandas import Series, DataFrame
import numpy as np 

fser100 = Series(np.arange(6))
fser200 = Series([90, 34, 67, 12, 100, 79])
fser300 = Series([50, 24, 56, 89, 33],
                 index=[0, 1, 2, 3, 5])
fser400 = Series(np.arange(5))

print(fser100)
print(fser200)
print(fser300)


fser100 + fser300
"""
0    50.0
1    25.0
2    58.0
3    92.0
4     NaN
5    38.0
dtype: float64
"""

fser400 + fser300
"""
0    50.0
1    25.0
2    58.0
3    92.0
4     NaN
5     NaN
dtype: float64
"""

