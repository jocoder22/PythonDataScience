#!/usr/bin/env python

import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np 

x = np.arange(0, 10, 0.01)
y = np.sin(x)

plt.plot(x, y)
plt.axhline(0.5, color='r')
plt.axhline(0.0, color='m', xmax=0.75, xmin=0.5)
plt.axvline(0.9, color='g', linestyle="--")
plt.axvline(4.0, color='r', linestyle="--", ymax=0.75, ymin=0.3)
plt.grid()

# Adding span

# adding lines
np.arange(1,10,0.1)


# plt.axhspan(-0.5,0.5, color='k', alpha=0.4)
# xmin=0.2 , xmax=0.8


# plt.grid()
# color='r'
# linestyle='-'

plt.show()
