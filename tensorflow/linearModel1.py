#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets

sp = '\n\n'
# plt.style.use('ggplot')

# 1: set objective
#  Assess how IQ, Years of Experience, level of Education, Gender, Age affects Income

# 2: collect necessary data
# Using stimulated dataset

np.random.seed(299)
xx1 = np.random.normal(120, 10, 4000).astype('int')
xx2 = np.random.normal(12, 2.5, 4000).astype('int')
xx3 = np.random.choice(3, 4000)
xx4 = np.random.normal(1,4000)
xx5 = np.random.normal(40, 6, 4000).astype('int')
DateB = np.datetime64('2018-01-30') - 365 * xx5

intercept_y = 3
err = np.random.normal(0, 1.1, 4000)

print(data.info(), data.shape, data.describe(), sep=sp)

yy = np.array([0.6*x1 + 2.3*x2 + 0.83*x3 + 2.2*x4 + 
                1.3*xx5 + intercept_y + err for x1,x2,x3,x4,x5 in
                zip(xx1,xx2,xx3,xx4,xx5)])

cols=['IQ', 'YearsExperience', 'levelEducation', 'Gender', 'DateBirth']

data = pd.DataFrame(list(zip(xx1, xx2, xx3, xx4, DateB)))
data['Income'] = yy
