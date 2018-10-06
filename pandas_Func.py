import pandas as pd 
from pandas import Series, DataFrame
import numpy as np 

fser100 = Series(np.arange(6))
fser200 = Series([90, 34, 67, 12, 100, 79])
fser300 = Series([50, 24, 56, 89, 33],
                 index=[0, 1, 2, 3, 5])
fser400 = Series(np.arange(5))
fser500 = Series([2,3,5,6,4],index=[0,1,2,3,5])

print(fser100)
print(fser200)
print(fser300)
print(fser400)
print(fser500)


# Arithmetics;
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

fser300 * fser500
"""
0    100
1     72
2    280
3    534
5    132
dtype: int64
"""

fser300 ** fser500
"""
0            2500
1           13824
2       550731776
3    496981290961
5         1185921
dtype: int64
"""

