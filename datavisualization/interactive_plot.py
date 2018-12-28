#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

x1 = np.linspace(1, 10, 100)
x2 = np.arange(0, 11)
y2 = [i for i in abs((np.cos(x2-1) * 326))]
plt.plot(x1, x1**2, label='Quadratic Function')
plt.plot(x1, x1**3, label='Cubic Function')
plt.bar(x2, y2, label='Cosine Function')
plt.legend()
plt.scatter(x1, 200 + np.sin(x1+2), label='Sine Function')
