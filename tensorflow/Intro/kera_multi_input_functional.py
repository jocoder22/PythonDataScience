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


# define input layer, for first dataset
# tf parameters must be ndarray
inputlayer1 = tf.keras.Input(shape=(_nshape1[1],))

# Model the first dataset
# using functional API, define first hidden layer for first dataset
dense1 = tf.keras.layers.Dense(60, activation='relu')(inputlayer1)


# Add dropouts
dropout = tf.keras.layers.Dropout(0.2)(dense1)

# Add dense2, the second dense layer for first dataset
dense22 = tf.keras.layers.Dense(31, activation='relu')(dropout)

# Add dropouts
dropout = tf.keras.layers.Dropout(0.1)(dense22)

# Add dense2, the second dense layer for first dataset
dense2 = tf.keras.layers.Dense(12, activation='relu')(dropout)



# define input layer, for second dataset
# tf parameters must be ndarray
# inputlayer22 = constant(data_reg.values, float)
inputlayer2 = tf.keras.Input(shape=(_nshape2[1],))

# Model the second dataset
# using functional API, define first hidden layer for second dataset
dense1b = tf.keras.layers.Dense(80, activation='relu')(inputlayer2)

# Add dropouts
dropout1 = tf.keras.layers.Dropout(0.2)(dense1b)

# Add dense2, the second dense layer for first dataset
dense22b = tf.keras.layers.Dense(35, activation='relu')(dropout1)

# Add dropouts
dropout2 = tf.keras.layers.Dropout(0.1)(dense22b)

# Add dense2b, the second dense layer for second dataset
dense2b = tf.keras.layers.Dense(12, activation='relu')(dropout2)
# dense2b = tf.keras.layers.Dense(3, activation='softmax')(dense22b)


# Merge model outputs
merged = tf.keras.layers.add([dense2, dense2b])

merged = tf.keras.layers.Dense(1)(merged)

# binary classification, use sigmoid activation
# merged = tf.keras.layers.Dense(1, activation='softmax')(merged)

# Define functional model
# pass in input tensor and merged output layer
model = tf.keras.Model(inputs=[inputlayer1, inputlayer2], outputs=merged)


# # Compile the model
# model.compile('adam', loss='categorical_crossentropy')
model.compile('adam', loss='mse', metrics=['mae'])

# # Print a model summary
print(model.summary())

# Add the number of epochs and the validation split
history = model.fit([data1, data2], targets_r, epochs=500, validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())



plt.plot(hist['epoch'], hist['mae'], label='Train Error_mae')
plt.plot(hist['epoch'], hist['val_mae'], label='Validation Error_mae')
plt.plot(hist['epoch'], hist['loss'], label='Train loss')
plt.plot(hist['epoch'], hist['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 200])
plt.title('Loss curve')
plt.legend()
plt.show()