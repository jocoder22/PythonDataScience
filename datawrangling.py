import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 2")


# Read data1
data1 = pd.read_csv("AAPL-Close-15-17.csv", index_col="date")
data1.head()
plt.plot(data1)
plt.show()


# Read data2
data2 = pd.read_csv("AAPL-EPS-15-17.csv", index_col="Date")
data2.head()
plt.plot(data2)
plt.show()

# Merging datas - Inner, Outer, Right, Left
# Pandas use join() method
# Note the indexes - date - of both dataset are not of the same format
data1.index   # '2015-01-02', '2015-01-05', '2015-01-06'
data2.index   #  '4/3/2017', '12/30/2016', '9/26/2016', '6/27/2016'


# convert to common datetime format
pd.to_datetime(data1.index)
pd.to_datetime(data2.index)

