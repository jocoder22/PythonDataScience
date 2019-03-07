#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
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
DateB = np.datatime64('2018-01-30') - 365 * xx5

intercept = 3
error = 

yy = np.random.normal(120, 10, 4000).astype('int')
