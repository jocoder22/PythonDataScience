import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from printdescribe import print2

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.facecolor'] = "0.92"

tf.reset_default_graph()


# Let's start by initialising two contants, called matrix1 and matrix 2
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

print2('matrix1:', matrix1, 'matrix2', matrix2)

# Create a matrix multiplication op
product = tf.matmul(matrix1, matrix2)

print2(product)

# Launch graph in a session
with tf.Session() as sess:
    result = sess.run(product)

# After the session is closed, all TensorFlow tensors and ops cease to exist
# result is actually a NumPy array and therefore persists after the session is closed
print2("")
print2(result)
print2(type(result))

with tf.Session() as sess:
    print2('matrix1:', matrix1.eval(),'matrix2', matrix2.eval())


# Here is another way of starting a session
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = x - a
print2(sub.eval())

sess.close()