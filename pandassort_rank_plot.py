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

data.sort_values(by='Box')
data.sort_values(by=['Box','Pen'])