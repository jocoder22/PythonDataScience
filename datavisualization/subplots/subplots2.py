#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


y = np.arange(10)
z = np.random.rand(len(y), 6)*1000

print(z)
fig, axes = plt.subplots(2,3, sharex=True, sharey=True)
axesr = axes.ravel()
print(axesr)

for idx in range(len(axesr)):
    axesr[idx].bar(y, z[:,idx], color=plt.cm.Paired(idx/6.))
    axesr[idx].set_title(f'Plot_{idx}')

plt.show()