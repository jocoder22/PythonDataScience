#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from collections import defaultdict

def preprocessor(inputfile, labelName, noise=True, batch=0):
    # data_1 = pd.read_csv(inputfile)
    
    # namelist = list(data_1.columns)
    # featureNames = namelist.remove(labelName)
    # label_idx = data_1.columns.get_loc(labelName)
    w = [[0.1]] * 4
    def decodecsv(row, noise=noise):
        parserow = tf.decode_csv(row, w)
        label = parserow[-1:]
        del parserow[-1]
        features = parserow

        if noise:
            features = td.add(features, td.random_normal(shape=[len(features)],
                                                         mean=0. ,
                                                         stddev=0.02))
            
        return features, label

    dataset = (tf.data.TextLineDataset(inputfile).skip(1).map(decodecsv))

    if batch > 0: 
        dataset = dataset.batch(batch)
            

    iterrr = dataset.make_one_shot_iterator()
    features, label = iterrr.get_next()

    return features, label


sp = '\n\n'
plt.style.use('ggplot')

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tensorflow\\'
os.chdir(path)


tensorIn = preprocessor(r'Array\tensorData.csv', 'Income', noise=False, batch=10)

with tf.Session() as ss:
    x, y = ss.run(tensorIn)
    print(x, y)


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






data = pd.read_csv(r'Array\tensofData.csv')
kk = [data.columns]

ppp = dict([(k, pd.Series(v)) for k, v in zip(kk, datav)])

mydict = defaultdict(list)
for element in datav:
    for idx, member in enumerate(element):
        mydict[kk[idx]].append(member)


df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in mydict.items()]))
df2 = pd.DataFrame.from_dict(mydict, orient='index').transpose()

