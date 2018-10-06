import pandas as pd 
from pandas import Series, DataFrame 
import numpy as np 

# create DataFrame;
data1 = DataFrame(np.arange(24).reshape(8, 3),
                  columns=["Age", "Height", "Weight"])

data1 - data1.loc[:,["Age","Weight"]]
data1.mean()
data1.std()

# for standardization;
(data1 - data1.mean())/data1.std()
data1.sum()


# Vectorization;
np.sqrt(data1)