#!/usr/bin/env python

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gfl

x1 = np.arange(0, 10, 0.1)
y1 = np.cos(x1)

fig, plx = plt.subplots(4, figsize=(20, 6), dpi=100, sharex=True)
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.2, right=0.89)

for i, xa in enumerate(plx):
    xa.scatter(x1, y1, c=np.sin(x1), s=np.power(x1, i))
    pos = list(xa.get_position().bounds)
    xpos = pos[0] - 0.01
    ypos = pos[1] + pos[3]/2.
    fig.text(xpos, ypos,'For power {}'.format(i), color='red', 
             va='center', ha='right', fontsize=11)
    xa.yaxis.set_ticks_position('right')

plt.show()
