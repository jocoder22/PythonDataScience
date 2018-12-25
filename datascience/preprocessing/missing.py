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


# Using sklearn Imputer
imput = Imputer(missing_values="NaN", strategy="mean", axis=0)
imput_fit =  imput.fit(mydata_pd.values)
Imputed_mydata = imput_fit.transform(mydata_pd.values)
print(Imputed_mydata)


# other strategy: median, most_frequent

data3 = np.random.randint(20, 35, size=88)
data3[random.sample([i for i in range(88)], 7)] = 999
data3[random.sample([i for i in range(88)], 7)] = 900
data4 = DataFrame(data3.reshape(22, 4), 
                  columns=['Age', 'Height', 'Weight', 'Grade'])

print(data4)


# Using sklearn Imputer II
imput2 = Imputer(missing_values=999, strategy="mean", axis=0)
imput_fit2 =  imput2.fit(data4.values)
Imputed_mydata2 = imput_fit.transform(data4.values)
print(Imputed_mydata2)