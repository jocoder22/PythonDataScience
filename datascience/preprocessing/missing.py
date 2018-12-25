#!/usr/bin/env python

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import Imputer
import random


mydata2 = np.random.uniform(low=12, high=100, size=(240,))
mydata2[random.sample([i for i in range(240)], 70)] = np.nan
mydata_pd = DataFrame(mydata2.reshape(60, 4), 
                  columns=['Age', 'Height', 'Weight', 'Grade'])
print(mydata_pd)
