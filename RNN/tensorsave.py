#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import sklearn
from datetime import datetime, date
import tensorflow as tf

import talib as tb
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, Dropout, Flatten
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


import pandas_datareader as pdr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = r'C:\Users\Jose\Desktop\PythonDataScience\RNN\model33'   
os.chdir(path)
savedir = os.path.join(os.getcwd(), 'model33')


"""
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')

saver = tf.train.Saver()



sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, savedir)
  
"""


# This will save following files in Tensorflow v >= 0.11
# my_test_model.data-00000-of-00001
# my_test_model.index
# my_test_model.meta
# checkpoint


with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    saver = tf.train.import_meta_graph('model33.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    print(sess.run('w1:0'))



path = r'C:\Users\Jose\Desktop\PythonDataScience\RNN\model33\model44'   
os.chdir(path)


""" 
savedir2 = os.path.join(os.getcwd(), 'model')

#Prepare to feed input, i.e. feed_dict and placeholders
w11 = tf.placeholder("float", name="w11")
w21 = tf.placeholder("float", name="w21")
b11 = tf.Variable(2.0, name="bias")
feed_dict ={w11:4, w21:8}
 
#Define a test operation that we will restore
w31  = tf.add(w11,w21)
w41 = tf.multiply(w31, b11,name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
#Create a saver object which will save all the variables
saver = tf.train.Saver()
 
#Run the operation by feeding input
print(sess.run(w41, feed_dict))
#Prints 24 which is sum of (w11+w21)*b11 
 
#Now, save the graph
saver.save(sess, savedir2)


"""
 
with tf.Session() as sess:    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    
    
    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data
    
    graph = tf.get_default_graph()
    w11 = graph.get_tensor_by_name("w11:0")
    w21 = graph.get_tensor_by_name("w21:0")
    feed_dict ={w11:13.0, w21:17.0}
    
    #Now, access the op that you want to run. 
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    
    print(sess.run(op_to_restore, feed_dict))
    #This will print 60 which is calculated 
    #using new values of w1 and w2 and saved value of b1.

