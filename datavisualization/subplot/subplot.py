#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# import matplotlib
# matplotlib.matplotlib_fname()
# display the location of matplotlibrc file

######set subplot params
# plt.rcParams['figure.subplot.wspace'] = 0.8

# grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

xplot = np.linspace(0, 10, 1000)
yplot = np.sin(xplot)

grid = plt.GridSpec(2, 3)
plt.tight_layout()
plt.subplot(grid[0,0])
plt.title('This is for cmap: Accent')
plt.scatter(xplot, yplot, c=np.cos(xplot), cmap='Accent',
            edgecolors='none',
            s=np.power(xplot, 2))
plt.subplot(grid[0,1])
plt.title('This is for cmap: CMRmap')
plt.scatter(xplot, yplot, c=np.cos(xplot), cmap='CMRmap',
            edgecolors='none',
            s=np.power(xplot, 3))

plt.subplot(grid[1,:2])
plt.title('This is for cmap: Set1_r')
plt.scatter(xplot, yplot, c=np.cos(xplot), cmap='Set1_r',
            edgecolors='none',
            s=np.power(xplot, 4))

plt.subplot(grid[:,2:])
plt.title('This is for cmap: Paired')
plt.scatter(xplot, yplot, c=np.cos(xplot), cmap='Paired',
            edgecolors='none',
            s=np.power(xplot, 4))

plt.show()