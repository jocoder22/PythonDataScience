import pandas as pd 
from pandas import Series, DataFrame
import numpy as np 

fser100 = Series(np.arange(6))
fser200 = Series([90, -34, 67, 12, -100, 79])
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

# Boolean operations; Compares identically index series
fser100 > fser200
fser400 > Series([1,1,0,2,5],index=[0,1,2,3,4])

np.abs(fser200)
"""
0     90
1     34
2     67
3     12
4    100
5     79
dtype: int64
"""
type(np.abs(fser200))  # <class 'pandas.core.series.Series'>

np.sqrt(np.abs(fser200))
"""
0     9.486833
1     5.830952
2     8.185353
3     3.464102
4    10.000000
5     8.888194
dtype: float64
"""

@np.vectorize
def trunc(x):
    return x if x > 0 else 0

trunc(np.array([-1,5,6,-3,0]))
# array([0, 5, 6, 0, 0])
trunc(fser100)
# array([0, 1, 2, 3, 4, 5])
trunc(fser200)
# array([90,  0, 67, 12,  0, 79], dtype=int64)