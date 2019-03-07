
#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

print(f'Python: {sys.version}')
print(f'Numpy: {np.__version__}')
print(f'Pandas: {pd.__version__}')

a = tf.constant(4.0, dtype=tf.float32)
b = tf.constant(5.0, dtype=tf.float32)
total = a+b

print(a)
print(b)
print(total)


write = tf.summary.FileWriter('Folder_1')
write.add_graph(tf.get_default_graph())

# on the conda prompt, navigate to the directory: cd C:\Users\okigboo\Desktop\PythonDataScience
# type: tensorboard --logdir=Folder_1

sess = tf.Session()
print(sess.run(total))

# for line equation; y = mx + b
# declare the constants and placeholder for the variables

m = tf.constant(9.0, name='x')
b = tf.constant(4.0, name='b')
x = tf.placeholder(dtype='float32', name='x')

# state the equation
y = m*x + b

print(y.eval({x: 8.0}, session=sess))


