#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# generate some data
x = np.arange(0, 100, 0.01)
# indices = range(0,10000, 101)
# x1 = np.take(x, indices)
# x1 = x[::10].copy
x1 = x[:1000]
y = np.sin(x)
cll = np.random.randn(len(x))
plt.style.use(['dark_background'])
# plot it
fig = plt.figure(figsize=(20, 6))
plt.tight_layout()
plt.subplot2grid((2, 3), (0, 0), colspan=2,  xticklabels=[],
                 xticks=[])
plt.plot(x, y, "g-.")

plt.subplot2grid((2, 3), (0, 2), rowspan=2)
plt.scatter(x1, np.sin(x1), c=np.cos(x1), cmap='Paired',
            edgecolors='none',
            s=np.power(x1, 2))

plt.subplot2grid((2, 3), (1, 0), colspan=2)
plt.scatter(x, y, c=cll, cmap='Accent',
            edgecolors='none',
            s=np.power(x, 1.5))
plt.show()