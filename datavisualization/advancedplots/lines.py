#!/usr/bin/env python

import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np 

x = np.arange(0, 10, 0.01)
y = np.sin(x)

plt.plot(x, y)

# adding lines
plt.axhline(0.5, color='r')
plt.axhline(0.0, color='m', xmin=0.5, xmax=0.75)
plt.axvline(0.9, color='g', linestyle="--")
plt.axvline(4.0, color='r', linestyle="--", ymin=0.3, ymax=0.75,)

# different axis parameters, can use axis='both' for both axis of the grid
plt.grid(color='c', linestyle='-.', axis='y')
plt.grid(color='y', linestyle=':', axis='x')

# Adding span
plt.axhspan(-0.15,0.25, color='k', alpha=0.4)
plt.axhspan(-0.75,-0.5, color='y', alpha=0.6, xmin=0.3, xmax=0.8)


plt.show()
