import numpy as np
import pandas as pd
from pandas import Series, DataFrame

series1 = Series([12, 23, 45, 21, 27, 31, 33])
series2 = Series(['Adam', 'Eve', 'kelly', 'Ben', 'Mary', 'John'])

print(series1)
print(series2)


# creating index;
indx = pd.Index(["USA", "Canada", "Algeria"
                 "Mexico", "Japan", "Kenya"
                 "Malaysia", "Holland", "Poland"
                 "Brazil", "South Korea", "China"])

print(indx)
type(indx)  # <class 'pandas.core.indexes.base.Index'>