#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from tensorflow import Variable, float32, keras, constant
# import tensorflow as tf
import matplotlib.pyplot as plt
# plt.style.use('ggplot')


path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = '\n\n'

# Read in the dataset, 
data = pd.read_csv('car.csv', compression='gzip')
data_reg = data.drop(columns=['Origin','Model_year'])
data_class = data.drop(columns=['Model_year'])

print(data_class.groupby('Origin')['MPG'].mean())

# Define the sequential model
model = keras.Sequential()

# Define the dimensions of the data
# Use the flatten() on the ndarray
_nshape = data_reg.values.flatten().shape

# Define the first dense hidden layer
model.add(keras.layers.Dense(200, activation='relu', input_shape = _nshape))

# Define the second dense hidden layer
model.add(keras.layers.Dense(25, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(1))



