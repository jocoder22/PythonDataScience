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

# define input layer, for regression
# tf parameters must be ndarray
inputlayer_reg = constant(data_reg.iloc[1:].values, float)

# Add dense1, the first dense layer
dense1 = keras.layers.Dense(18, activation='relu')(inputlayer_reg)

# Add dense2, the second dense layer
dense2 = keras.layers.Dense(6, activation='relu')(dense1)

# Add output layer, the final layer
output = keras.layers.Dense(1)(dense2)


#############################################################################
# Define binary Origin
data_class2 = data_class.copy()
data_class2['Origin'] = data_class2['Origin'].map({1:0, 2:1, 3:1})
print(data_class2.groupby('Origin')['MPG'].mean())

targets_b = data_class2.pop("Origin")

# define input layer, for binary category
# tf parameters must be ndarray
inputlayer_class1 = constant(data_class2.values, float)

# Add dense1, the first dense layer
dense1b = keras.layers.Dense(18, activation='relu')(inputlayer_class1)

# Add dense2, the second dense layer
dense2b = keras.layers.Dense(6, activation='relu')(dense1b)

# Add output layer, the final layer
output_b = keras.layers.Dense(1, activation='sigmoid')(dense2b)



#######################################################################
# Define multiclass Origin
data_class3 = data_class.copy()
data_class3['Origin'] = data_class3['Origin'].map({1:0, 2:1, 3:2})
print(data_class3.groupby('Origin')['MPG'].mean())

targets_c = data_class3.pop("Origin")
# define input layer, for binary category
# tf parameters must be ndarray
inputlayer_class2 = constant(data_class3.values, float)

# Add dense1, the first dense layer
dense1c = keras.layers.Dense(50, activation='relu')(inputlayer_class2)

# Add dropouts
dropout1 = keras.layers.Dropout(0.2)(dense1c)

# Add dense2, the second dense layer
dense2c = keras.layers.Dense(20, activation='relu')(dropout1)

# Add output layer, the final layer
output_c = keras.layers.Dense(3, activation='softmax')(dense2c)
