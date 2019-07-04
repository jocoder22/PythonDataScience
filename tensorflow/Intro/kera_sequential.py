#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from tensorflow import Variable, float32, keras, constant
# import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
plt.style.use('ggplot')

path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = '\n\n'

# Read in the dataset, 
data = pd.read_csv('car.csv', compression='gzip')
data_reg = data.drop(columns=['Origin','Model_year'])
data_class = data.drop(columns=['Model_year'])
targets_r = data_reg.pop('MPG')
print(data_class.groupby('Origin')['MPG'].mean())

# Define the sequential model
model = keras.Sequential()

# Define the dimensions of the data
# Use the flatten() on the ndarray
_nshape = data_reg.values.shape[1]

# Define the first dense hidden layer
model.add(keras.layers.Dense(200, activation='relu', input_shape=(_nshape,)))

# Define the second dense hidden layer
model.add(keras.layers.Dense(25, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(1))


# Compile the model
model.compile('adam', loss='mse')


# Print a model summary
print(model.summary())




#############################################################################
# Define binary Origin
data_class2 = data_class.copy()
data_class2['Origin'] = data_class2['Origin'].map({1:0, 2:1, 3:1})
print(data_class2.groupby('Origin')['MPG'].mean())

targets_b = data_class2.pop("Origin")

# Define the sequential model
model2 = keras.Sequential()

# Add the first dense hidden layer
model2.add(keras.layers.Dense(300, activation='relu', input_shape=(_nshape,)))

# Add the second dense layer
model2.add(keras.layers.Dense(100, activation='relu'))

# Add output layer, the final layer
model2.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model2.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
# model2.compile('rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Print a model summary
print(model2.summary())


#######################################################################
# Define multiclass Origin
data_class3 = data_class.copy()
data_class3['Origin'] = data_class3['Origin'].map({1:0, 2:1, 3:2})
print(data_class3.groupby('Origin')['MPG'].mean())

targets_c = data_class3.pop("Origin")

_nshape = data_class3.values.shape[1]

# Add one-hot encoding
yy = keras.utils.to_categorical(targets_c)
print(targets_c.head(), yy[11:25], sep=sp, end=sp)

# """
# Normalize the dataset
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data_class3)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, targets_c, test_size=0.2, stratify=targets_c)



# Define the sequential model
model3 = keras.Sequential()

# Add the first dense hidden layer
model3.add(keras.layers.Dense(600, activation='relu', input_shape=(_nshape,)))

# Add the second dense layer
model3.add(keras.layers.Dense(300, activation='relu'))

# # Add the second dense layer
# model3.add(keras.layers.Dense(60, activation='relu'))

# Add output layer, the final layer
model3.add(keras.layers.Dense(3, activation='softmax'))


sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile the model
# model3.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
model3.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Print a model summary
print(model3.summary())

history = model3.fit(X_train, y_train,
                    batch_size=100, epochs=50,
                    # validation_data=(X_test, y_test)
                    validation_split=0.2)

score = model3.evaluate(X_test, y_test)
print(f'\n\nTest Loss: {score[0]}\nTest Accuracy: {score[1]}')

ypred = model3.predict_classes(X_test)
print(ypred)
commatrix = confusion_matrix(y_test, ypred)

cm = multilabel_confusion_matrix(y_test, ypred)

print(commatrix, cm, end=sp, sep=sp)

# Draw heatmap
classes = 'USA Europe Japan'.split()
sns.heatmap(commatrix, annot=True, fmt='d', cbar=False, linewidths=.5, cmap="YlGnBu")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks(np.arange(3) + 0.5, classes, rotation=45)
plt.yticks(np.arange(3) + 0.5, classes)
plt.show()

# define classification_report
print(classification_report(y_test, ypred, target_names=classes))