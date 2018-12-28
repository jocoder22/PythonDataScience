#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt

# plt.ion()

x1 = np.arange(0, 4, 0.1)
plt.plot(x1, x1, label='Linear')
plt.plot(x1, np.sin(x1), label='sin(x)')
plt.plot(x1, x1**2, label='Quadratic')
plt.plot(x1, np.cos(x1), label='cos(x)')


plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('My graphic display I')
plt.legend()

plt.show()