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


m,n = mnist.train.images.shape
number_to_show = 100

def show_digits(i=0):
    """
    Show some of the digits
    """
    im = np.reshape(mnist.train.images[i], (28,28))
    plt.imshow(im, cmap='viridis') 
    plt.title('The digits')
    
    plt.show()
    
w = interact(show_digits, i =(0, number_to_show)) 

im = np.reshape(mnist.train.images[0], (28,28))
plt.imshow(im, cmap='viridis') 
plt.title('The digits')
plt.show()

n_inputs = 28 * 28   # The size of each image
n_outputs = 10     # There are 10 digits, and therefore 10 classes
tf.reset_default_graph()

stddev = 2/np.sqrt(n_inputs)   

with tf.name_scope("variable"):
    W = tf.Variable(tf.truncated_normal((784,10), stddev=stddev), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    
with tf.name_scope("placeholder"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_") 

with tf.name_scope("output"):
    logits = tf.nn.softmax(tf.matmul(x,W) + b, name="logits")
    Y_prob = tf.nn.softmax(logits, name="Y_prob")


with tf.name_scope("train"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_, name="xentropy")
    loss = tf.reduce_mean(xentropy, name='loss')
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, name="train_op")
    
with tf.name_scope("eval"):
    correct = tf.equal(tf.argmax(logits,axis=1), tf.argmax(y_,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    
with tf.name_scope("init_and_save"):
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()  # We need to add a saver Op

# Now we add averything we'll need in future to a collection
tf.add_to_collection('train_var', train_op)
tf.add_to_collection('train_var', accuracy)
tf.add_to_collection('train_var', x)
tf.add_to_collection('train_var', y_)


n_epoch = 100  

with tf.Session() as sess:
    sess.run(init_op)
    
    graph = tf.get_default_graph()
    print2(graph.get_name_scope())

    for epoch in range(n_epoch):   
        
        # One step of the training
        sess.run(train_op, feed_dict={x: mnist.train.images, 
                                         y_: mnist.train.labels})
        
        # Evaluate accuracy on the training set
        acc_train = accuracy.eval(feed_dict={x: mnist.train.images, 
                                             y_: mnist.train.labels})
        
        # Evaluate the accuracy on the test set. This should be 
        # smaller than the accuracy on the training set
        acc_test  = accuracy.eval(feed_dict={x: mnist.test.images,
                                             y_: mnist.test.labels})
        
        print(epoch, "Train accuracy:", acc_train, ",  Test accuracy:", acc_test)   
    
    # Print the variable names for later access.
    var = [v.name for v in tf.trainable_variables()]
    print(var) 
    
    weights = graph.get_tensor_by_name("variable/W:0").eval()
    bias  = graph.get_tensor_by_name("variable/b:0").eval()
    
    # Save the current values of the variables, the meta graph, and the collection
    # variables to files.
    save_path = saver.save(sess,"./test/test.ckpt")  
    saver.export_meta_graph(filename='./test/test.meta',
                            collection_list=["train_var"])
    
    print2(graph.get_name_scope())
                            
print2('weights:', weights)
print2('bias:', bias)


m,n = weights.shape

def show_weights(i=0):
    im = weights.T[i].reshape([28,28])
    plt.imshow(im, cmap='viridis') 
    plt.title('The weigths of filter '+str(i))
    plt.show()
    
w = interact(show_weights, i =(0, n-1)) 
im = weights.T[0].reshape([28,28])
plt.imshow(im, cmap='viridis') 
plt.title('The weigths of filter '+str(0))
plt.show()


# get the name scopes and operations
graph = tf.get_default_graph()
print2(graph.get_name_scope(), graph.get_all_collection_keys())
