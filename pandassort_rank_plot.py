import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame 


data = DataFrame(np.round(np.random.randn(7,3) *12),
                 columns=['Box', 'Pen', 'Books'],
                 index=list('defcagb'))
print(data)


# sorting the index;
data.sort_index()
data.sort_index(axis=1)

# sorting the columns with axis=1;
data.sort_index(axis=1, ascending=False)


# Sorting by values;
data.sort_values(by='Box')
data.sort_values(by=['Box','Pen'])
data.sort_values(by='Books')
data.sort_values(by=['Books','Box'])


# ranking show the ranking of values if they are sorted;
data.rank() # methods for breaking ties: Average(default), max, min, first
data.rank(method='max')
data.rank(method='first')
data.rank(method='min')