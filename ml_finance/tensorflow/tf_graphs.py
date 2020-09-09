#!/usr/bin/env python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import shutil
# import tensorflow as tf
import tensorflow.compat.v1 as tf


from printdescribe import print2, changepath
from datetime import datetime

tf.disable_v2_behavior()
print2(" ")

path22 = r"D:\PythonDataScience"
sys.path.insert(0, path22)
import input_data

patht =  r"D:\PythonDataScience\ml_finance"
os.chdir(patht)
# print2(os.getcwd())


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = f"{root_logdir}/run-{now}/"
print2(logdir)


for file in os.scandir(os.path.join(f"{root_logdir}")):
    # if file.name.endswith(".bak"):
    # os.unlink(file.path)
    print(file)
    shutil.rmtree(file)


# filelist = [ f for f in os.listdir(mydir) if f.endswith(".bak") ]
# for f in filelist:
#     os.remove(os.path.join(mydir, f))

# Generate data
x_train = np.linspace(0, 1, 100) 
y_train =  0.45 * x_train + 8.89 + 0.21 * np.random.randn(x_train.shape[0])

tf.reset_default_graph()

learning_rate = 0.001

# Model parameters (variables). Now organised under the same named scope
# It is important to name the variables; this is what will be used in tensorboard
with tf.name_scope('variables') as scope:
    w = tf.Variable([-3.], dtype=tf.float32, name = 'w')
    b = tf.Variable([3.], dtype=tf.float32, name = 'b')


# Model input (placeholders)
with tf.name_scope('placeholders') as scope:
    x = tf.placeholder(tf.float32, name = 'x')
    y = tf.placeholder(tf.float32, name = 'y')


# Linear model
with tf.name_scope('model') as scope:
    linear_model = w * x + b
    mse = tf.reduce_sum(tf.square(linear_model - y), name = 'mse') 
    grad = tf.gradients(mse, [w, b], name = 'grad')


# Gradient descent update
with tf.name_scope('training') as scope:
    update_w = tf.assign(w, w - learning_rate * grad[0], name = 'update_w')
    update_b = tf.assign(b, b - learning_rate * grad[1], name = 'update_b')


# Define init Op
with tf.name_scope('init_op') as scope:
    init_op = tf.global_variables_initializer()


# Define a saver Op that will allow us to save the graph
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Execution
with tf.Session() as sess:
    sess.run(init_op)
    graph = tf.get_default_graph()
    
    # 1000 steps
    for step in range(1000):
        # Report loss every 100 steps
        if step%100 == 0:
            summary_str = mse_summary.eval(feed_dict={x: x_train, y: y_train})
            file_writer.add_summary(summary_str, step)
        sess.run([update_w, update_b], {x:x_train, y:y_train})

file_writer.close()


# Access to the variables in a graph
for op in tf.get_default_graph().get_operations():
    print(op.name)