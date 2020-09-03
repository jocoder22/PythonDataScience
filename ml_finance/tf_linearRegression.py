import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from printdescribe import print2
print2(" ")

# Generate data
x_train = np.linspace(0, 1, 800)
y_train = 0.2789 * x_train + 8.84 + 0.01 * np.random.randn(x_train.shape[0])

plt.plot(x_train, y_train, 'r.')
plt.title('Generated Data for Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()