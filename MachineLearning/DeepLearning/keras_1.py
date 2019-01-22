# Import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import losses
plt.style.use('ggplot')

url= 'https://assets.datacamp.com/production/repositories/654/datasets/8a57adcdb5bfb3e603dad7d3c61682dfe63082b8/hourly_wages.csv'

df = pd.read_csv(url, sep=',')
print(df.columns)

predictors = df.drop('wage_per_hour', axis=1)
target = df['wage_per_hour']

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target, epochs=20)


# This is for categorical outcome
# download titanic dataset
url = 'https://assets.datacamp.com/production/repositories/502/datasets/e280ed94bf4539afb57d8b1cbcc14bcf660d3c63/titanic.csv'
titan = pd.read_csv(url, sep=',')

mapper = {'Q': 0, 'S': 1,'C': 2, 'female': 0, 'male': 1}

for x in ['sex','embarked']:
    titan[x] = titan[x].map(mapper)

titan.fillna(0, inplace=True)
to_drop = [s for s in titan if titan[s].dtype == 'object']
to_drop.append('survived')
predictors2 = titan.drop(to_drop, axis=1).values
n_cols = predictors2.shape[1]

print(n_cols)



early_stop_monitor = EarlyStopping(patience=3)
# Convert the target to categorical: target
target2 = to_categorical(titan.survived)


# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors2, target2, validation_split=0.3, epochs=40, callbacks=[early_stop_monitor])

# # Calculate predictions: predictions
# predictions = model.predict(X_test)

# # Calculate predicted probability of survival: predicted_prob_true
# predicted_prob_true = predictions[:,1]

# # print predicted_prob_true
# print(predicted_prob_true)







