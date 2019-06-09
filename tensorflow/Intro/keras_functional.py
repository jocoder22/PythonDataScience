#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from tensorflow import Variable, float32, keras, constant
import tensorflow as tf
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

# define constant variables
_nshape = data_reg.values.shape
data = constant(data_reg.values, float32)
targets_r = constant(targets_r.values, float32)

# define input layer, for binary category
# tf parameters must be ndarray
inputlayer_ = tf.keras.Input(shape=(_nshape[1],))


# Add dense1, the first dense layer
dense1 = keras.layers.Dense(18, activation='relu')(inputlayer_)

# Add dense2, the second dense layer
dense2 = keras.layers.Dense(6, activation='relu')(dense1)

# Add output layer, the final layer
output = keras.layers.Dense(1)(dense2)

model = keras.Model(inputs=[inputlayer_], outputs=output)

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())


#############################################################################
# Define binary Origin
data_class2 = data_class.copy()
data_class2['Origin'] = data_class2['Origin'].map({1:0, 2:1, 3:1})
print(data_class2.groupby('Origin')['MPG'].mean())

targets_b = data_class2.pop("Origin")

# define constant variables
_nshape2 = data_class2.values.shape
data2 = constant(data_class2.values, float32)
targets_b = constant(targets_b.values, float32)

# define input layer, for binary category
# tf parameters must be ndarray
inputlayer_class1 = tf.keras.Input(shape=(_nshape2[1],))

# Add dense1, the first dense layer
dense1b = keras.layers.Dense(18, activation='relu')(inputlayer_class1)

# Add dense2, the second dense layer
dense2b = keras.layers.Dense(6, activation='relu')(dense1b)

# Add output layer, the final layer
output_b = keras.layers.Dense(1, activation='sigmoid')(dense2b)

model2 = keras.Model(inputs=[inputlayer_class1], outputs=output_b)

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


# define constant variables
_nshape3 = data_class3.values.shape
data3 = constant(data_class3.values, float32)
targets_c = constant(targets_c.values, float32)

# define input layer, for binary category
# tf parameters must be ndarray
inputlayer_class2 = tf.keras.Input(shape=(_nshape3[1],))

# Add dense1, the first dense layer
dense1c = keras.layers.Dense(50, activation='relu')(inputlayer_class2)

# Add dropouts
dropout1 = keras.layers.Dropout(0.2)(dense1c)

# Add dense2, the second dense layer
dense2c = keras.layers.Dense(20, activation='relu')(dropout1)

# Add output layer, the final layer
output_c = keras.layers.Dense(3, activation='softmax')(dense2c)

model3 = keras.Model(inputs=[inputlayer_class2], outputs=output_c)

# Compile the model
model3.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model3.summary())


