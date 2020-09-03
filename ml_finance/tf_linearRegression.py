import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from printdescribe import print2
print2(" ")

# set matplotlib parameters
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.facecolor'] = "0.92"

# set random seed
np.random.seed(3)

# Generate data
x_train = np.linspace(0, 1, 100) 
y_train =  0.2 * x_train + 8.89 + 0.21 * np.random.randn(x_train.shape[0])

plt.plot(x_train, y_train, 'r.')
plt.title('Generated Data for Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# reset default graph
tf.reset_default_graph()

# Set hyperparameter
learning_rate = 0.001

# Model parameters (variables). 
# Initialised with values not close to the correct values.
w = tf.Variable([-3.9], dtype=tf.float32)
b = tf.Variable([3.6], dtype=tf.float32)

# Model input (placeholders)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Linear model
linear_model = w * x + b

# loss, sum of squares errors
mse = tf.reduce_sum(tf.square(linear_model - y)) 

# Gradient, with multiple outputs
grad = tf.gradients(mse, [w, b])

# Gradient descent update
# update the weights and bias
update_w = tf.assign(w, w - learning_rate * grad[0])
update_b = tf.assign(b, b - learning_rate * grad[1])

# Define init Op
init_op = tf.global_variables_initializer()

# Execution
with tf.Session() as sess:
    sess.run(init_op)

    # Descend for 1000 steps
    for i in range(1000):
        # print loss at every 100th step
        if i%100 == 0:
            print('mse = ', sess.run(mse, feed_dict = {x:x_train, y:y_train}))
            
        sess.run([update_w, update_b], feed_dict = {x:x_train, y:y_train})

    # evaluate training accuracy
    curr_W, curr_b, curr_mse = sess.run([w, b, mse], feed_dict = {x:x_train, y:y_train})
    print(f"w: {curr_W} b: {curr_b} loss: {curr_mse}")