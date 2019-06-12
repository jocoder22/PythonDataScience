#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
# plt.style.use('ggplot')


path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
path2 = r'C:\Users\Jose\Desktop\PythonDataScience\tensorflow\Intro'
os.chdir(path)
sp = '\n\n'

# load the dataset, 
data = pd.read_csv('car.csv', compression='gzip')
# print(data.groupby('Model_year')['MPG'].count(), end=sp)


# Define target
target = data.pop('MPG')


# define dummies for categorical variables
# data = pd.get_dummies(data, columns=['Origin','Model_year','Cylinders'], prefix='D', 
#                     drop_first=True)

# from numpy import unique
# print(unique(data.Model_year).shape)
# print(target.shape)

# colnames = data.columns
print(data.head())
data = pd.get_dummies(data, columns=['Origin'], prefix='Dummy', 
                    drop_first=True)
colnames = data.columns
scaler =  StandardScaler()
data2 = scaler.fit_transform(data)
lasso = Lasso(alpha=0.1)

lass_coef = lasso.fit(data2, target).coef_

_ = plt.plot(range(len(colnames)), lass_coef)
_ = plt.xticks(range(len(colnames)), colnames)
_ = plt.ylabel('Coefficients')
plt.show()



# data = pd.get_dummies(data, columns=['Origin'], prefix='D', 
#                     drop_first=True)

# Initialize and fix the scaler
# scaler =  StandardScaler()
# scaler2 = MinMaxScaler()
# data2 = scaler.fit_transform(data)
data = pd.DataFrame(data2, columns=data.columns)

# split the data to tran and test
xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2, random_state=45)


# Define the datasets
# xtrain2 = pd.DataFrame([xtrain.pop(x) for x in xtrain.columns[0:4]]).T
# xtest2 = pd.DataFrame([xtest.pop(x) for x in xtest.columns[0:4]]).T
xtrain2 = pd.DataFrame([xtrain.pop(x) for x in xtrain.columns[0:6]]).T
xtest2 = pd.DataFrame([xtest.pop(x) for x in xtest.columns[0:6]]).T
# print(xtrain.shape, xtrain2.shape,  sep=sp, end=sp)

# define constant variables
# data1 = tf.constant(data_class.values, tf.float32)
# data1 = tf.constant(xtrain.values, tf.float32)
# data2 = tf.constant(xtrain2.values, tf.float32)

# xtest = tf.constant(xtest.values, tf.float32)
# xtest2 = tf.constant(xtest2.values, tf.float32)


# # Define target tensors
# ytrain = tf.constant(ytrain.values, tf.float32)
# ytest = tf.constant(ytest.values, tf.float32)

# ytrain = tf.constant(ytrain.values, tf.float32)
# ytest = tf.constant(ytest.values, tf.float32)

# Define the dimension of the 2 datasets
_nshape1 = xtrain.values.shape
_nshape2 = xtrain2.values.shape


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
inputlayer2 = tf.keras.Input(shape=(_nshape2[1], ))

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

# Add output1 layers
# output1 = tf.keras.layers.Dense(1)(dense2b)

# Merge model outputs
merged = tf.keras.layers.add([dense2, dense2b])

merged = tf.keras.layers.Dense(3, activation='relu')(merged)
merged = tf.keras.layers.Dense(1)(merged)

# binary classification, use sigmoid activation
# merged = tf.keras.layers.Dense(1, activation='softmax')(merged)

# Define functional model
# pass in input tensor and merged output layer
# model = tf.keras.Model(inputs=[inputlayer1, inputlayer2], outputs=[merged, output1])
model = tf.keras.Model(inputs=[inputlayer1, inputlayer2], outputs=merged)


# # Compile the model
# model.compile('adam', loss='categorical_crossentropy')
model.compile('adam', loss='mse', metrics=['mae'])

# # Print a model summary
print(model.summary())

# Plot the model
os.chdir(path2)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
graph = plt.imread('model.png')
plt.figure(figsize=(5, 5))
plt.imshow(graph)
plt.axis('off')
plt.show()


# Add the number of epochs and the validation split
# history = model.fit([data1, data2], [ytrain, ytrain], epochs=500, steps_per_epoch=20)
# history = model.fit([xtrain, xtrain2], [ytrain, ytrain], epochs=200, validation_split=0.1)
history = model.fit([xtrain, xtrain2], ytrain, epochs=350, validation_split=0.1)
# colnames = 'loss output2_loss output1_loss output2_mae output1_mae'.split()

hist = pd.DataFrame(history.history)
# hist.columns = colnames
hist['epoch'] = history.epoch
print(hist.tail(), sep=sp, end=sp)


for x in hist.columns:
    if x != 'epoch':
        plt.plot(hist['epoch'], hist[x], label=x)
# plt.plot(hist['epoch'], hist['val_loss'], label='Validation Loss')
# plt.plot(hist['epoch'], hist['output1_loss'], label='First Output Loss')

# plt.plot(hist['epoch'], hist['loss'], label='Train Loss')
# plt.plot(hist['epoch'], hist['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 50])
plt.title('Loss curve')
plt.legend()
plt.show()


# Evaluate the model
loss, mae = model.evaluate([xtest, xtest2], ytest, verbose=0)
# loss, loss2, loss1, mae2, mae1 = model.evaluate([xtest, xtest2], [ytest, ytest], verbose=0)
print(f'Mean absolute error = {mae:.2f}')
# print(f'Mean absolute error = {mae2:.2f}')


