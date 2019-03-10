#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import plot_model
from sklearn import datasets
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')


path = r'C:\Users\Jose\Desktop\PythonDataScience\tensorflow\Array'
os.chdir(path)

sp = '\n\n'
data = pd.read_csv('train.csv')

# EDA
print(data.columns.tolist(), data.shape, sep=sp)

print(data.isnull().values.any(), end=sp)
print(data.isnull().sum().tolist(), end=sp)

#### show the rows index with missing values
print(data.isnull().index.tolist())
print(data.loc[data.isnull().sum(1)>1].index.tolist())
print(data.loc[data[['V4', 'V5', 'V6', 'V7', 'V8','V14', 'V15', 'V16','V17', 'V18', 'V19']].isnull().any(1)].index)

#### print the columns with missing values
print(data.columns[data.isnull().any()].tolist())
print(data.head(), data.shape, data.info(), sep=sp)
print(data[['V8','V14', 'V15', 'V16','V17', 'V18', 'V19']].tail(), end=sp)

data.dropna(subset=['V7', 'V17'], inplace=True)
print(data.isnull().values.any(), data.shape, data.info(), sep=sp)


# Get the features and labels
xd = data.drop(columns=['Class'])
ycat = data[['Class']]


# Initialize the scaler and OneHotEncoder
scaler = MinMaxScaler()
onehot = OneHotEncoder(sparse=False, categories='auto') # spare=False will return an np array

# tranform the features and label
xdata = scaler.fit_transform(xd)
x = np.array(xdata)
y = onehot.fit_transform(ycat)

print(x.shape)
print(y.shape, type(y), sep=sp)
print(y[:5])


# form pandas dataframe of y
ydata = pd.DataFrame(y, columns=['Fraud', 'NonFraud'])
print(ydata.head())

# define the model function
def tf_modeler(features):
    _nshape = features.shape[1]
    model_g = tf.keras.models.Sequential()
    model_g.add(tf.keras.layers.BatchNormalization(input_shape=(_nshape,)))
    model_g.add(tf.keras.layers.Dense(1000, activation='relu'))
    model_g.add(tf.keras.layers.Dropout(0.1))

    model_g.add(tf.keras.layers.BatchNormalization())
    model_g.add(tf.keras.layers.Dense(500, activation='relu'))
    model_g.add(tf.keras.layers.BatchNormalization())
    model_g.add(tf.keras.layers.Dropout(0.2))

   
    model_g.add(tf.keras.layers.Dense(250, activation='relu'))
    model_g.add(tf.keras.layers.BatchNormalization())
    model_g.add(tf.keras.layers.Dropout(0.1))
    
   
    model_g.add(tf.keras.layers.Dense(50, activation='relu'))
    model_g.add(tf.keras.layers.BatchNormalization())
    model_g.add(tf.keras.layers.Dropout(0.05))


    model_g.add(tf.keras.layers.Dense(2))
    model_g.add(tf.keras.layers.Activation('sigmoid'))
    model_g.compile(tf.keras.optimizers.Adam(lr=0.001), 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return model_g

model1= tf_modeler(x)
model1.fit(x, y, epochs=25, verbose=1, validation_split=0.2, batch_size=16)

plot_model(model1, to_file='model.png')

lossValues = pd.DataFrame(model1.history.history)
lossValues = lossValues.rename({'val_loss':'ValidationLoss',  'val_acc':'Val_Accuray', 
                            'loss':'TrainLoss', 'acc':'TrainAccuracy'}, axis='columns')

print(lossValues.head())
ValidationLoss = model1.history.history['val_loss']
Val_Accuray = model1.history.history['val_acc']
TrainLoss = model1.history.history['loss']
TrainAccuracy = model1.history.history['acc']

# plot loss values
plt.plot(ValidationLoss)
plt.plot(TrainLoss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve')
plt.legend(['Validation Loss', 'Train Loss'])
plt.show()


# plot Accuracy
plt.plot(Val_Accuray)
plt.plot(TrainAccuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Validation Accuracy', 'Train Accuracy'])
plt.show()
