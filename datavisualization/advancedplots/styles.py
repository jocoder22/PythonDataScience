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



ax1 = plt.subplot2grid((4,3), (0, 0), rowspan=2, colspan=2)
ax1.plot(x1, np.power(x1, 3))
ax1.xaxis.set_ticks_position('top')
ax2 = plt.subplot2grid((4,3), (0, 2), xticks=[])
ax2.scatter(x1, y1, c=np.sin(x1), s=np.power(x1, 2))
ax3 = plt.subplot2grid((4,3), (2, 0), rowspan=2)
ax3.hist(np.sin(x1))
ax4 = plt.subplot2grid((4,3), (2, 1))
ax4.hist(np.random.normal(size=1500), alpha=0.8)
ax4.hist(np.random.normal(1, size=800), alpha=0.6)
ax5 = plt.subplot2grid((4,3), (3, 1))
ax5.scatter(x1, np.tan(x1), c=np.sin(x1))
ax6 = plt.subplot2grid((4,3), (1, 2), rowspan=3)
ax5.scatter(x1, np.sin(x1), c=np.sin(x1), s=np.power(x1, 3))
ax6.scatter(x1, np.sin(x1), c=np.sin(x1), s=np.power(x1, 3))


plt.show()

