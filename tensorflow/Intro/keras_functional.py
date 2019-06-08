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

# define input layer
# tf parameters must be ndarray
inputlayer_reg = constant(data_reg.iloc[1:].values, float)

# Add dense1, the first dense layer
dense1 = keras.layers.Dense(18, activation='relu')(inputlayer_reg)

# Add dense2, the second dense layer
dense2 = keras.layers.Dense(6, activation='relu')(dense1)