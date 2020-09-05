#!/usr/bin/env python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import shutil
# import tensorflow as tf

from ipywidgets import interact


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from printdescribe import print2, changepath
from datetime import datetime
print2(" ")

path22 = r"D:\PythonDataScience"
sys.path.insert(0, path22)
import input_data

path2 = r"D:\Wqu_FinEngr\Machine Learning in Finance\CourseMaterials\Module5\WQU_MLiF_Module5_Notebooks\ML M5 Notebooks (updated)"

with changepath(path2):
    print2(os.getcwd())
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tf.reset_default_graph()

# Everything we need should be restored from what we saved before
# We do need an Op for restoration!
saver = tf.train.import_meta_graph("./test/test.ckpt.meta")

with tf.Session() as sess:
    
    # We restore the session and the default graph
    # is the previously saved graph
    saver.restore(sess,"./test/test.ckpt")
    graph = tf.get_default_graph()
    
    # Check that we recognise the variables saved before.
    var = [v.name for v in tf.trainable_variables()]
    print(var)
    
    # We can retrieve and display the previous weights.
    prev_weights = graph.get_tensor_by_name("variable/W:0").eval()
    prev_bias  = graph.get_tensor_by_name("variable/b:0").eval()
    
    # Retrieve all the Ops we need from the collection
    train_op = tf.get_collection("train_var")[0]
    accuracy = tf.get_collection("train_var")[1]
    x = tf.get_collection("train_var")[2]
    y_ = tf.get_collection("train_var")[3]

    # Now we simply continue the training
    for train_step in range(200):
        
        sess.run(train_op, feed_dict={x: mnist.train.images, 
                                      y_: mnist.train.labels})
        
        if train_step % 10 == 0:
            
            # Evaluate accuracy on the training set
            acc_train = accuracy.eval(feed_dict={x: mnist.train.images, 
                                             y_: mnist.train.labels})
        
            # Evaluate the accuracy on the test set. 
            acc_test  = accuracy.eval(feed_dict={x: mnist.test.images,
                                             y_: mnist.test.labels})
        
            print(train_step, "Train accuracy:", acc_train, ",  Test accuracy:", acc_test)   
        
        
    update_weights = graph.get_tensor_by_name("variable/W:0").eval()
    update_bias = graph.get_tensor_by_name("variable/b:0").eval()
    
    
# Checking that we have something useful
print("Weights:", prev_weights.shape)
print("Bias:", prev_bias)


m,n = update_weights.shape

def show_update_weights(i=0):
    """
    Show the weights
    """
    im = update_weights.T[i].reshape([28,28])
    plt.imshow(im, cmap='viridis') 
    plt.title('The weigths of filter '+str(i))
    
    plt.show()
    
w = interact(show_update_weights, i =(0, n-1))
im = update_weights.T[0].reshape([28,28])
plt.imshow(im, cmap='viridis') 
plt.title('The weigths of filter '+str(0))

plt.show()