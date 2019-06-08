#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from tensorflow import Variable, float32, keras
import tensorflow as tf
import matplotlib.pyplot as plt



# plt.style.use('ggplot')


path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = '\n\n'

# Read in the dataset, 
data = pd.read_csv('car.csv', compression='gzip')
data = data.drop(columns=['Origin','Model_year'])

# define the trainable variables
intercept = Variable(0.1, float32)
slope = Variable(0.1, float32)

# define loss function
def lossfunc(intercept, slope, feature, target):
    predictions = intercept + slope * feature
    return keras.losses.mse(target, predictions)


# Initialize the Adam optimizer
optim = keras.optimizers.Adam()


# Run the linear model
for batch in pd.read_csv('car.csv', compression='gzip', chunksize=150):
    feature_batch = tf.cast(batch['Horsepower'], float32)
    target_batch = tf.cast(batch['MPG'], float32)
    optim.minimize(lambda: lossfunc(intercept, slope, feature_batch, target_batch), var_list=[intercept, slope])
    print(intercept.numpy(), slope.numpy())



# Print trainable variables
print(intercept.numpy(), slope.numpy())
print(data.head())



