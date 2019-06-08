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
targets_r = data_reg.pop('MPG')
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

# Compile the model
model.compile('adam', loss='mse')

# Print a model summary
print(model.summary())




#############################################################################
# Define binary Origin
data_class2 = data_class.copy()
data_class2['Origin'] = data_class2['Origin'].map({1:0, 2:1, 3:1})
print(data_class2.groupby('Origin')['MPG'].mean())

targets_b = data_class2.pop("Origin")

# Define the sequential model
model2 = keras.Sequential()

# Add the first dense hidden layer
model2.add(keras.layers.Dense(300, activation='relu', input_shape=_nshape))

# Add the second dense layer
model2.add(keras.layers.Dense(100, activation='relu'))

# Add output layer, the final layer
model2.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model2.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model2.summary())


#######################################################################
# Define multiclass Origin
data_class3 = data_class.copy()
data_class3['Origin'] = data_class3['Origin'].map({1:0, 2:1, 3:2})
print(data_class3.groupby('Origin')['MPG'].mean())

targets_c = data_class3.pop("Origin")

# Define the sequential model
model3 = keras.Sequential()

# Add the first dense hidden layer
model3.add(keras.layers.Dense(600, activation='relu', input_shape=_nshape))

# Add the second dense layer
model3.add(keras.layers.Dense(100, activation='relu'))

# Add output layer, the final layer
model3.add(keras.layers.Dense(3, activation='softmax'))

# Compile the model
model3.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model3.summary())
