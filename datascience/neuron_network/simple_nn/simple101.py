#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
# # from tensorflow.keras.datasets import mnist
# import tensorflow as tf
# # # load mnist datasets
# (xtrain, ytrain),(xtest, ytest) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
# # (xtrain, ytrain),(xtest, ytest) = mnist.load_data(path='mnist.npz')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

fig = plt.figure(figsize=[20,20])
for i in range(6):
    ax = fig.add_subplot(1,6, i+1, xticks=[], yticks=[])
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(str(y_train[i]))
plt.show()

# show the first digit
plt.imshow(X_train[0], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title(str(y_train[0]))
plt.show()

# rescale [0,255] 
X_train = X_train.astype('float32')/255
X_train = X_test.astype("float32")/255


