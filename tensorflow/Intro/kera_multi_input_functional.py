#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# plt.style.use('ggplot')


path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = '\n\n'

# Read in the dataset, 
data = pd.read_csv('car.csv', compression='gzip')
print(data.groupby('Model_year')['MPG'].count())
data_class = data.loc[:,['Origin','Model_year']]
data_reg = data.iloc[:, 1:6]

df_dummy = pd.get_dummies(data_class, columns=['Origin','Model_year'], prefix='D', 
                    drop_first=True)



# define constant variables
# data1 = tf.constant(data_class.values, tf.float32)
data1 = tf.constant(df_dummy.values, tf.float32)
data2 = tf.constant(data_reg.values, tf.float32)
targets_r = tf.constant(data.pop('MPG').values, tf.float32)

# Define the dimension of the 2 datasets
# _nshape1 = data_class.values.shape
_nshape1 = df_dummy.values.shape
_nshape2 = data_reg.values.shape
