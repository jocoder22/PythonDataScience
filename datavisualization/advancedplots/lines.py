#!/usr/bin/env python

import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np 

x = np.arange(0, 10, 0.01)
y = np.sin(x)

plt.plot(x, y)

# adding lines
plt.axhline(0.5, color='r')
plt.axhline(0.0, color='m', xmax=0.75, xmin=0.5)
plt.axvline(0.9, color='g', linestyle="--")
plt.axvline(4.0, color='r', linestyle="--", ymax=0.75, ymin=0.3)
plt.grid(color='c', linestyle='-.')

# Adding span
plt.axhspan(-0.15,0.25, color='k', alpha=0.4)
plt.axhspan(-0.75,-0.5, color='y', alpha=0.6, xmax=0.8, xmin=0.3)


plt.show()
