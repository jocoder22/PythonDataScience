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