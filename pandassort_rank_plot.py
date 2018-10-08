import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame 


data = DataFrame(np.round(np.random.randn(7,3) *12),
                 columns=['Box', 'Pen', 'Books'],
                 index=list('defcagb'))
print(data)