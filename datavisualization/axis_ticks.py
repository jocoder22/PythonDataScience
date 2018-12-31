#!/usr/bin/env python

import matplotlib.pyplot as plt 
from matplotlib import ticker
import numpy as np 


x = np.linspace(0, 10, 1000)
y = np.sin(x)

plt.figure(figsize=(20, 60))
plt.tight_layout()

ax1 = plt.subplot(231)
plt.setp(ax1.get_xticklabels(), fontsize=12)
ax1.yaxis.set_ticks_position('right')
plt.plot(x, y)
plt.xlim(0.01, 8.0)

# sharex and sharey means sharing axis limit
ax2 = plt.subplot(232, yticks=[], sharex=ax1)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.plot(x, y)

ax3 = plt.subplot(233)
ax3.plot(x, y)
ax3.xaxis.set_ticks_position('top')
ax3.xaxis.set_major_locator(ticker.MaxNLocator(9))

# prune='upper''lower' 'both'
# integer=True
plt.xlim(0.01, 5.0)

plt.show()

