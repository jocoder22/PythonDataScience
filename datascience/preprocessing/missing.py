#!/usr/bin/env python

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import Imputer
import random

random.seed(2)
mydata2 = np.random.uniform(low=12, high=100, size=(240,))
mydata2[random.sample([i for i in range(240)], 70)] = np.nan
mydata_pd = DataFrame(mydata2.reshape(60, 4), 
                  columns=['Age', 'Height', 'Weight', 'Grade'])
print(mydata_pd)

print(mydata_pd.isnull().sum())

mydata_pd.values[:5, :]

# droping missing values
## drop missing row
mydata_pd.dropna(axis=0)

## drop missing features, ver bad idea
mydata_pd.dropna(axis=1)
