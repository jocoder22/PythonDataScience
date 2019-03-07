#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from datetime import datetime

sp = '\n\n'
# plt.style.use('ggplot')

# 1: set objective
#  Assess how IQ, Years of Experience, level of Education, Gender, Age affects Income

# 2: collect necessary data
# Using stimulated dataset

np.random.seed(123)
xx1 = np.random.normal(120, 10, 4000).astype('int')
xx2 = np.random.normal(12, 2.5, 4000)
xx3 = np.random.choice(4, 4000)
xx4 = np.random.choice(2,4000)
xx5 = np.random.normal(40, 6, 4000).astype('int')
DateB = np.datetime64('2018-01-30') - 365*xx5
race = np.random.choice(['Black', 'White', 'Asian', 'Chinese', 'Latino'], 4000)

intercept_y = 3
err = np.random.normal(0, 1.1, 4000)



yy = np.array([1.6*x1 + 2.3*x2 + 0.83*x3 + 1.2*x4 + 
                1.3*x5 + intercept_y + err for x1,x2,x3,x4,x5, err in
               zip(xx1, xx2, xx3, xx4, xx5, err)])

cols=['IQ', 'YearsExperience', 'levelEducation', 'Gender', 'DateBirth', 'Race']

data = pd.DataFrame(list(zip(xx1, xx2, xx3, xx4, DateB, race)), columns=cols)
data['Income'] = yy

print(data.info(), data.head(), data.shape, data.describe(), sep=sp)


# cleaning data
# drop null values
data.dropna(axis=0)
data = data[data.YearsExperience > 0]

print(data.info(), data.head(), data.shape, data.describe(), sep=sp, end=sp)
print(data.describe(include=['datetime64']))
