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
matrix11 = tf.constant([[23., 13.]])
matrix12 = tf.constant([[42.],[62.]])

print2('matrix11:', matrix11, 'matrix2', matrix12)

# Create a matrix multiplication op
product = tf.matmul(matrix11, matrix12)
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
    print2('matrix11:', matrix11.eval(),'matrix12', matrix12.eval())


# Here is another way of starting a session
sess = tf.InteractiveSession()

xx = tf.Variable([31.0, 20.0])
aa = tf.constant([43.0, 83.0])

# Initialize 'xx' using the run() method of its initializer op.
xx.initializer.run()

# Add an op to subtract 'aa' from 'xx'. Run it and print the result
substract = xx - aa
print2(substract.eval())

sess.close()


# Variables and placeholders
# Typically, you will feed your training or test data into 
# placeholders during execution, using a dictionary feed.
# reset default graph
tf.reset_default_graph()

# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an update Op.
update = tf.assign(state, state + 1)

# Variables must be initialized by running an `init` Op after having launched the graph.  
# We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()

# Launch the graph and run the ops.
with tf.Session() as sess:
    
  # Run the 'init' op. 
  sess.run(init_op)
    
  # Print the initial value of 'state'
  print2(state.eval())
    
  # Run the op that updates 'state' and print 'state'. Note that the graph is executed several times.
  for _ in range(3):
    sess.run(update)
    print2(state.eval())