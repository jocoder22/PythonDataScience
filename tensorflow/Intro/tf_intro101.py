#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from tensorflow import Variable, float32
import tensorflow as tf
import matplotlib.pyplot as plt


# plt.style.use('ggplot')


path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = '\n\n'

# Read in the dataset, 
data = pd.read_csv('car.csv', compression='gzip')
data = data.drop(columns=['Origin','Model_year'])

# define the variables
intercept = Variable(0.1, float32)
slope = Variable(1.0, float32)

# define loss function
def lossfunc(intercept, slope, feature, target):
    predictions = intercept + slope * feature
    return tf.keras.losses.mse(target, predictions)


# Run the linear model
for batch in pd.read_csv('car.csv', compression='gzip', chunksize=150):
    feature = tf.cast(batch['Horsepower'], float32)
    targets = tf.cast(batch['MPG'], float32)
    opt.minimize()

print(data.head())