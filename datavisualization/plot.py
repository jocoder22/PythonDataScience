#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

xplot = np.linspace(0, 10, 1000)
yplot = np.sin(xplot)

plt.plot(xplot, yplot)
