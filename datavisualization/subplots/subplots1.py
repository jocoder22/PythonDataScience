#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


y = np.arange(10)
z = np.random.rand(len(y), 6)*1000

fig, axes = plt.subplots(2,3, sharex=True, sharey=True)

for idx, ax in enumerate(axes.flatten()):
    ax.bar(y, z[:,idx], color=plt.cm.Paired(idx/6.))

plt.show()