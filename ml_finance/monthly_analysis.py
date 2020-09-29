#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pandas_datareader as pdr
import  datetime
import calendar

from printdescribe import print2, changepath

import warnings
warnings.filterwarnings('ignore')

start_date = '2000-01-01'
symbol = "C"  # "^GSPC" "GOOG"
data = pdr.get_data_yahoo(symbol, start_date)[['Adj Close']]

print2(data.head())
ts = data.copy()


# ts.groupby([ts.index.month,ts.index.year]).mean().unstack().plot(figsize=(12,8))
# plt.show()

# plot average weekday or monthly overall periods
# ts.groupby([ts.index.month]).min().unstack().plot(figsize=(12,8))
# plt.xticks(np.arange(5), 'Monday Tuesday Wednesday Thursday Friday'.split())
# ts.groupby([ts.index.month]).max().unstack().plot(figsize=(12,8))
# plt.legend(["Min", "Mean", "Max"])

ts.groupby([ts.index.weekday]).mean().unstack().plot(figsize=(12,8))
dayname  = [calendar.day_name[x] for x in range(0,5)]
plt.xticks(np.arange(5), dayname)
plt.grid()
plt.show()

ts['month'], ts["year"] = ts.index.month, ts.index.year

# month1 = ts[ts.month == 2]
# month1.loc[:, ['Adj Close', 'year']].groupby(['year']).mean().plot()
# plt.show()
# print2(ts.head(), month1.head())


groups = ts.groupby([ts.index.month, ts.index.year]).mean()


# groups.index = ["Datemonth", "Dateyear"]
# fig, ax = plt.subplots()
# # ax.set_color_cycle(colors)
# for name, group in groups:
# #     ax.plot(group.index.month, group["Adj Close"], marker='o', linestyle='', ms=12, label=name)
# # ax.legend(numpoints=1, loc='upper left')
#     print(name, group)

# plt.show()
# print2(groups)


fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
# ax.margins(0.05)
for idx, ax in enumerate(ax.flatten(),start=1):
    # monthname = datetime.date(1900,  idx, 1).strftime('%B')
    monthname = calendar.month_name[idx]
    month1 = ts[ts.month == idx]
    ax.plot(month1.groupby(['year'])["Adj Close"].min(), label=monthname)
    ax.set_xlabel(monthname)
    ax.legend()
plt.show()


print2(ts.head())
yy = ts.groupby(['year','month'], as_index=False).mean()
month1 = yy[yy.month == idx]
print2(yy.head(), month1.head())


fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
# ax.margins(0.05)
for idx, ax in enumerate(ax.flatten(),start=1):
    monthname = calendar.month_name[idx]
    month1 = yy[yy.month == idx]
    ax.plot(month1.year, month1["Adj Close"], label=monthname)
    ax.legend()
plt.show()


monthname2 = [calendar.month_name[x] for x in range(1,13)]
yy2 = ts.groupby(['year','month'], as_index=False).mean()
y3 = yy2.groupby(["month"], as_index=False)["Adj Close"].mean()
y3["monthname"] = y3.month.apply(lambda x: calendar.month_name[x])
plt.plot(y3["monthname"], y3["Adj Close"])
plt.grid()
plt.show()
print2(y3)

dayname  = [calendar.day_name[x] for x in range(0,5)]
print2(monthname2,dayname)
