#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# change current working directory
path = r'D:\PythonDataScience\datascience\neuron_network'
os.chdir(path)

modelname = 'Best.h5'
savedir = os.path.join(os.getcwd(), 'weights')
if not os.path.isdir(savedir):
    os.makedirs(savedir)
filepath = os.path.join(savedir, modelname)

sp = {"end":"\n\n", "sep":"\n\n"}

# # from tensorflow.keras.datasets import mnist
# # # load mnist datasets
# (xtrain, ytrain),(xtest, ytest) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
# # (xtrain, ytrain),(xtest, ytest) = mnist.load_data(path='mnist.npz')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# show some of the images
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
X_test = X_test.astype("float32")/255

# one hot encode the labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# build our model
model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(250, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(10, activation="softmax"))

# summary of our model
model.summary()

# compile our model
model.compile(loss="categorical_crossentropy", 
             optimizer='rmsprop',
             metrics=['accuracy'])

# save model
# modelname = 'BestModel.h5'
# filepath2 = os.path.join(os.getcwd(), modelname)
# model.save(filepath2)
       

# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose = 0)
accuracy = score[1] * 100

print(f'Test accuracy : {accuracy:.2f}')

# only comment the lines below after training our model
# checkpoint = ModelCheckpoint(filepath=filepath, verbose=1,
#                             save_best_only=True)

# hist = model.fit(X_train, y_train, batch_size=128,
#                 epochs=10, validation_split=0.2,
#                 callbacks=[checkpoint],
#                 verbose=1, shuffle=True)


# plt.figure(figsize=[10,8])
# plt.plot(hist.history['loss'], label="Train Loss")
# plt.plot(hist.history['val_loss'], label="Validation Loss")
# plt.legend()
# plt.show()


# loading model
# first load the model
# then secondly, add the weights
# model = load_model("BestModel.h5")
   
# to load only the weights you must laod the model as above,and then run
# model.load_weights('weights\Best.h5')
# model.fit with callbacks save only the weights

model.load_weights(r'weights\Best.h5')

# # evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose = 0)
accuracy = score[1] * 100

print(f'Test accuracy : {accuracy:.2f}')



