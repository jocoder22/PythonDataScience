#!/usr/bin/env python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import shutil
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from printdescribe import print2, changepath
from datetime import datetime
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



path2 = r"D:\Wqu_FinEngr\Machine Learning in Finance\CourseMaterials\Module5\WQU_MLiF_Module5_Notebooks\ML M5 Notebooks (updated)"


with changepath(path2):
    print2(os.getcwd())
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


from ipywidgets import interact

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
print("Weights:",weights.shape)
print("Bias:",bias)


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