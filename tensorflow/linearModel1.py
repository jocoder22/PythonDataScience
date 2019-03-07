#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn import datasets
from datetime import datetime


sp = '\n\n'
plt.style.use('ggplot')

# 1: set objective
#  Assess how IQ, Years of Experience, level of Education, Gender, Age affects Income

# 2: collect necessary data
# Using stimulated dataset

np.random.seed(123)
xx1 = np.random.normal(120, 10, 4000).astype('int')
xx2 = np.random.normal(12, 2.5, 4000)
xx3 = np.random.choice(4, 4000)
xx4 = np.random.choice(2,4000)
xx5 = np.random.normal(40, 6, 4000).astype('int')
DateB = np.datetime64('2018-01-30') - 365*xx5
race = np.random.choice(['Black', 'White', 'Asian', 'Chinese', 'Latino'], 4000)

intercept_y = 3
err = np.random.normal(0, 1.1, 4000)



yy = np.array([1.6*x1 + 2.3*x2 + 
                1.3*x5 + intercept_y + err for x1,x2,x5, err in
               zip(xx1, xx2, xx5, err)])

cols=['IQ', 'YearsExperience', 'levelEducation', 'Gender', 'DateBirth', 'Race']

data = pd.DataFrame(list(zip(xx1, xx2, xx3, xx4, DateB, race)), columns=cols)
data['Income'] = yy

print(data.info(), data.head(), data.shape, data.describe(), sep=sp)


# cleaning data
# drop null values
data.dropna(axis=0)
data = data[data.YearsExperience > 0]
data['Race'] = data.Race.astype('category')
print(data.info(), data.head(), data.shape, data.describe(), sep=sp, end=sp)


# EDA
print(data.describe(include=['datetime64', 'category']))
pd.plotting.scatter_matrix(data, figsize=(10, 60),diagonal='kde')
plt.show()


sns.heatmap(data.corr(), annot=True, cbar=False, cmap='coolwarm')  # "YlGnBu" , 'coolwarm'
plt.xticks(rotation=0)
plt.show()


# Data processing;
data['Age'] = data.DateBirth.apply(
    lambda x: (datetime.strptime('2018-01-30', '%Y-%m-%d') - x).days/365)

data = data[np.logical_and(data.Age > 21.0 , data.Age < 60.0)]
print(data.info(), data.head(), data.shape, data.describe(), sep=sp, end=sp)



# Train and Evaluate Models
# X = data.drop(['DateBirth', 'Race', 'Income'], axis=1)
X=data[['IQ', 'YearsExperience', 'Age']]
y = data['Income']

data_idx = X.sample(frac=0.70).index
Xtrain = X[X.index.isin(data_idx)].values
Xtest = X[~X.index.isin(data_idx)].values


ytrain = y[y.index.isin(data_idx)].values
ytest = y[~y.index.isin(data_idx)].values

tf.reset_default_graph()

sess = tf.Session()

print(X.columns)

# Create parameter
weight_ = tf.get_variable(name='w', 
                          initializer=[[0.01], [0.01],  [0.01]])
interc_ = tf.get_variable(name='b', initializer=0.0)

# create input placeholders
x_ = tf.placeholder('float32', name='x')
y_ = tf.placeholder('float32', name='y')


# create linear model
yhat = tf.reshape(tf.matmul(x_, weight_) + interc_, [-1, ], name='yhat')

# create the loss and test score functions
mse = tf.reduce_mean(tf.square(y_ - yhat), name='mse')
rmse = tf.sqrt(mse, name='rmse')

# create test score
nrmse = tf.divide(rmse, tf.abs(tf.reduce_mean(y_)), name='nrmse')

# intialise variables
init = tf.variables_initializer([weight_, interc_])
sess.run(init)

# Training and Evaluation
# reset parameters of weight_ and interc_ 
opt = tf.train.GradientDescentOptimizer(learning_rate=0.0011)
train = opt.minimize(rmse)

for i in range(10000):
    if (i % 100 == 0):
        nrmse_result = sess.run(nrmse, {x_:Xtrain, y_: ytrain})
        print(f'Test NRMSE: {nrmse_result}')
    else:
        # sess.run([summaries, train], {x_:Xtrain, y_:ytrain})
        sess.run(train, {x_: Xtrain, y_: ytrain})

print(sess.run([weight_, interc_]))

