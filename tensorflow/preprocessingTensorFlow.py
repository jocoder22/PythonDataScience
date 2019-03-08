#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn import datasets
from datetime import datetime


def preprocessor(inputfile, featureNames, labelName, noise=True, batch=0):
    data_1 = pd.read_csv(inputfile)
    labelName = labelName
    namelist = list(data_1.columns)
    featureNames = namelist.remove(labelName)
    label_idx = data_1.columns.get_loc(labelName)
    w_ = [[0.1]] * len(namelist)
    def decodecsv(row, noise=noise):
        parserow = tf.decode_csv(row, w_):
        label = parserow[label_idx]
        del parserow[label_idx]
        features = parserow

        if noise:
            features = td.add(features, td.random_normal(shape=[len(features)],
                                                         mean=0. ,
                                                         stddev=0.02))
            dict_ , label_ = dict(zip(featureNames, features)), label
        return features, label












sp = '\n\n'
plt.style.use('ggplot')

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tensorflow\\'
os.chdir(path)


# load numpy array
Xtrain = np.load(r'Array\Xtrain.npy')
Xtest = np.load(r'Array\Xtest.npy')
ytrain = np.load(r'Array\ytrain.npy')
ytest = np.load(r'Array\ytest.npy')


# splitting datasets, either as tuple or dictionary
trainingData = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
trainingData2 = tf.data.Dataset.from_tensor_slices({'xtrainset': Xtrain, 'ytrainset': ytrain})

print(trainingData2.output_types, trainingData2.output_shapes, sep=sp)


# options: TFRecordDataset, TextLineDataset

# Iterators:
iterr = trainingData2.make_one_shot_iterator()
next_ele = iterr.get_next()

# Start a session
sess = tf.Session()

for i in range(len(Xtrain) + 3):
    try:
        result = sess.run(next_ele)
        print(result['xtrainset'])
    except tf.errors.OutOfRangeError as e:
        print(f'Out of Range: {i}')
