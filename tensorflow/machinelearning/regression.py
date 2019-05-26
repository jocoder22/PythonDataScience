#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests 
# from sklearn import datasets
# from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.preprocessing import Imputer
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import seaborn as sns
# plt.style.use('ggplot')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


sp = '\n\n'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
urlname = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names'
urlindex = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/Index"


# for i in [urlname, urlindex, url]:
#     respond = requests.get(i)
#     text = respond.text
#     print(text, end=sp)


colname = '''MPG Cylinders Displacement Horsepower Weight Acceleration 
             Model_year Origin Car_name'''.split()

origin_name = ['USA', 'Europe', 'Japan']

# mytypes = ({cname: float if cname != 'Car_name' else str for cname in colname})
# mytypes = ({cname: str for cname in colname})

# dataset = pd.read_csv(url, names=colname, dtype=mytypes, na_values=['na','Na', '?'],
#                 skipinitialspace=True, comment='\t', sep=" ", quotechar='"')


dataset = pd.read_csv(url, names=colname, na_values=['na','Na', '?'],
                skipinitialspace=True, comment='\t', sep=" ", quotechar='"')



dataset.drop(columns='Car_name', inplace=True)
print(dataset.isna().sum())
dataset.dropna(inplace=True)
dataset2 = dataset.copy()
data4 = dataset.copy()
# print(dataset.head(), dataset.shape, dataset.info(), sep=sp, end=sp)


encoder = OneHotEncoder(categorical_features = [7])
dataset = encoder.fit_transform(dataset).toarray()

newcolumns = colname[:-2]
origin_name.extend(newcolumns)
dataset = pd.DataFrame(dataset, columns=origin_name)

# print(dataset.head(), dataset.shape, dataset.info(), sep=sp)

originmap = {1.0 : 'USA', 2.0: 'Europe', 3.0: 'Japan'}
dataset2['Origin'] = dataset2['Origin'].map(originmap)
dataset2["Origin"] = dataset2["Origin"].astype('category')
# newcolumns.extend(['Europe', 'Japan', 'USA'])

data3 = pd.get_dummies(dataset2, columns=['Origin'])
# print(data3.head(), data3.shape, sep=sp, end=sp)



# data4['Origin'] = data4['Origin'].astype(int)
dumnames = ['USA', 'Europe', 'Japan']
for idx, ele in enumerate(dumnames, 1):
    data4[ele] = (data4['Origin'] == idx) * 1.0

# data4.drop(columns="Origin", inplace=True)
data4["Origin2"] = data4["Origin"].astype('category')
data4.drop(columns="Origin", inplace=True)
data5 = data4.copy()
data4j = data4.copy()
# print(data4.head(), data4.info(), sep=sp, end=sp)


# Exploratory data analysis
fig, ax = plt.subplots()
toplot = []
for features in data4.columns:
    toplot.append(data4[features])

ax.boxplot(toplot)
ax.set_xticklabels(data4.columns)
plt.show()

fig, ax = plt.subplots(ncols=data4.shape[1])

for idx, ax in enumerate(ax.flatten()):
    ax.boxplot(data4.iloc[:, idx])
    ax.set_xlabel(data4.columns[idx])
plt.show()



data4.drop(columns=["Origin2","Model_year", "USA", "Europe", "Japan"], inplace=True)
print(data4.info(), data4.shape, sep=sp)
scaler2 = MinMaxScaler()
data4sb = scaler2.fit_transform(data4)
data4s = pd.DataFrame(data4sb, columns=data4.columns)
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.set_title('Before Scaling')
ax1.set_xticklabels(data4.columns, rotation=90)
ax2.set_title('After MinMax Scaler')
for features in data4.columns:
    sns.kdeplot(data4[features].values, ax=ax1, label=features)
    sns.kdeplot(data4s[features].values, ax=ax2, label=features)
plt.legend()
plt.show()


sns.pairplot(data4, diag_kind='kde')
plt.show()



# define the model function
def tf_modeler(features):
    _nshape = features.shape[1]
    model_g = tf.keras.models.Sequential()
    model_g.add(tf.keras.layers.BatchNormalization(input_shape=(_nshape,)))
    model_g.add(tf.keras.layers.Dense(1000, activation='relu'))
    model_g.add(tf.keras.layers.Dropout(0.5)) # 0.4

    model_g.add(tf.keras.layers.BatchNormalization())
    model_g.add(tf.keras.layers.Dense(500, activation='relu'))
    model_g.add(tf.keras.layers.BatchNormalization())
    model_g.add(tf.keras.layers.Dropout(0.2)) # 0.5
    
   
    model_g.add(tf.keras.layers.Dense(50, activation='relu'))
    model_g.add(tf.keras.layers.BatchNormalization())
    model_g.add(tf.keras.layers.Dropout(0.2))


    model_g.add(tf.keras.layers.Dense(1))
    model_g.compile(tf.keras.optimizers.Adam(lr=0.001), 
                    loss='mse',
                    metrics=['mae', 'mse']) # 95.38

    return model_g



# # split the dataset
xtrain, xtest, ytrain, ytest = train_test_split(data5.iloc[:, 1:-1], data5.iloc[:, 0], 
                                test_size=0.2, random_state=45, stratify=data5.Origin2)

scaler =  StandardScaler()
xtrainscaled = scaler.fit_transform(xtrain)
xtestscaled = scaler.fit_transform(xtest)


def regmodel():
    modeler = keras.Sequential([
        # layers.Dense(124, activation=tf.nn.relu, input_shape=[len(xtrainscaleddf.keys())]),
        layers.Dense(124, activation=tf.nn.relu, input_shape=[xtrainscaled.shape[1]]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(16, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.01)

    modeler.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    
    return modeler


model = regmodel()

print(model.summary())

examaples = xtrainscaled[:10]
results1 = model.predict(examaples)
print(results1)


class PrintDD(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0 : print('Next 100 ')
        print("...", end='')
        

Epochs = 1000
