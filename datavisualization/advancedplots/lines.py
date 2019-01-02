#!/usr/bin/env python

import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np 

x = np.arange(0, 10, 0.01)
y = np.sin(x)

plt.plot(x, y)



plt.show()
