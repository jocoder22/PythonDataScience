#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pandas_datareader as pdr

from printdescribe import print2, changepath

import warnings
warnings.filterwarnings('ignore')

start_date = '2000-01-01'
data = pdr.get_data_yahoo("^GSPC", start_date)[['Adj Close']]

print2(data.head())
ts = data.copy()
# ts.groupby([ts.index.month,ts.index.year]).mean().unstack().plot(figsize=(12,8))
# plt.show()

# ts.groupby([ts.index.month]).min().unstack().plot(figsize=(12,8))
# ts.groupby([ts.index.month]).mean().unstack().plot(figsize=(12,8))
# ts.groupby([ts.index.month]).max().unstack().plot(figsize=(12,8))
# plt.legend(["Min", "Mean", "Max"])
# plt.grid()
# plt.show()

ts['month'], ts["year"] = ts.index.month, ts.index.year
month1 = ts[ts.month == 2]
month1.loc[:, ['Adj Close', 'year']].groupby(['year']).mean().plot()
plt.show()
print2(ts.head(), month1.head())

groups = nn.groupby('max_feature')
fig, ax = plt.subplots()
# ax.set_color_cycle(colors)
ax.margins(0.05)
for name, group in groups:
    ax.plot(group.learning_rate, group.max_depth, marker='o', linestyle='', ms=12, label=name)
ax.legend(numpoints=1, loc='upper left')

plt.show()

