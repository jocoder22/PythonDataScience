import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import losses
from keras.optimizers import SGD
plt.style.use('ggplot')

url = 'https://assets.datacamp.com/production/repositories/654/datasets/24769dae9dc51a77b9baa785d42ea42e3f8f7538/mnist.csv'

df = pd.read_csv(url, sep=',')
df.shape

X = df.iloc[:,1:]
target = df.iloc[:,0]
y = pd.get_dummies(target)
n_cols = X.shape[1]
# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))


# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.add(Dense(1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

early_stop_monitor = EarlyStopping(patience=3)

# Fit the model
model.fit(X, y, validation_split=0.3, epochs=90, callbacks=[early_stop_monitor])
