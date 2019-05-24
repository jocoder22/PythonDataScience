#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


x = np.arange(10)
y = np.random.rand(len(x), 6)*400

fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
flataxes = axes.flatten()
print(flataxes)

for idx, ax in enumerate(axes.flatten()):
    ax.bar(x, y[:,idx], color=plt.cm.Paired(idx/6.))

plt.show()