import pandas as pd 
from pandas import Series, DataFrame 
import numpy as np 
import random

# create DataFrame;
data1 = DataFrame(np.arange(24).reshape(8, 3),
                  columns=["Age", "Height", "Weight"])

data1 - data1.loc[:,["Age","Weight"]]
data1.mean()
data1.std()
data1.mean(axis='columns')


# for standardization;
(data1 - data1.mean())/data1.std()
data1.sum()


# Vectorization;
myobj = {'Baby': [10, 90, 40],
         'Girl': [22, 32, 25],
         'Boy': [54, 56, 67], 
         'Woman': ['Pen', 'Book', 'kett'],
         'Man': [89, 90, 78]}

data2 = DataFrame(myobj)
data2.mean
data2.mean(axis='columns')


np.sqrt(data1)
# Mixed data type problem;
np.sqrt(data2)  # error: all column must be numeric
# use the select_dtypes() method to select only numeric;
np.sqrt(data2.select_dtypes([np.number]))

@np.vectorize
def trunc(x):
    return x if x > 12 else 0

@np.vectorize
def trunc2(x):
    return x if x > 30 else 0

trunc(data1)
# Mixed data type problem
trunc2(data2)  # Error, must all be of same type
trunc2(data2.select_dtypes([np.number]))

# function application;
def geometricMean(dff):
    return dff.prod() ** (1 / len(dff))

geometricMean(data1)
data1.apply(geometricMean, axis='columns')

data1.applymap(lambda x: x if x > 12 else 12)




# Function chaining;
data1.apply(lambda x : x ** 0.5)
data1.select_dtypes([np.number]).apply(lambda x : x ** 0.5)



# Handling missing data;
# generate data;
data3 = np.random.randn(44)
data3[random.sample([i for i in range(44)], 7)] = np.nan
data4 = DataFrame(data3.reshape(11, 4), 
                  columns=['Age', 'Height', 'Weight', 'Grade'])

print(data4)


mser = Series([23, 45, 56.0, 90, np.nan, 55, 67, np.nan, 78])
print(mser)

np.isnan(data4)
data4.isnull()
data4.notnull()

# drop missing data;
data4.dropna()
print(mser.dropna())
mser2 = mser.copy()

# default dropna() does not work in place
# to work in place (change the data permanently) 
# use set inplace option to true
mser2.dropna(inplace=True)
