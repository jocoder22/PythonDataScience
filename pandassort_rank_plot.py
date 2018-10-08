import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from pandas import Series, DataFrame


data = DataFrame(np.round(np.random.randn(7, 3) * 12),
                 columns=['Box', 'Pen', 'Books'],
                 index=list('defcagb'))
print(data)


# sorting the index;
data.sort_index()
data.sort_index(axis=1)

# sorting the columns with axis=1;
data.sort_index(axis=1, ascending=False)


# Sorting by values;
data.sort_values(by='Box')
data.sort_values(by=['Box','Pen'])
data.sort_values(by='Books')
data.sort_values(by=['Books','Box'])


# ranking show the ranking of values if they are sorted;
data.rank()  # methods for breaking ties: Average(default), max, min, first
data.rank(method='max')
data.rank(method='first')
data.rank(method='min')


# Plotting;
seplot = Series(np.random.randn(200), columns='Debts')
dplot = DataFrame(np.random.randn(1000, 3).cumsum(axis=0),
                  columns=['Price', 'Discount', 'Sales'])
seplot1 = dplot.Price
type(seplot1)  # <class 'pandas.core.series.Series'>

dplot.head()
dplot.tail()
dplot.shape


# Plot series;
plt.plot(seplot)
plt.show()
seplot1.plot(kind='line', ylim=(-30, 48))
plt.show()

seplot1.hist()
plt.show

seplot1.plot(kind='hist')
plt.show

seplot1.plot(kind='kde')
plt.show


# Plot dataframe;
dplot.plot(kind='line')
plt.show()

dplot.plot(kind='box')
plt.show()
dplot.plot(kind='scatter', x='Price', y='Sales')
plt.show()
dplot.plot(kind='hexbin', x='Price', y='Sales')
plt.show()
dplot.plot(kind='hexbin', x='Price', y='Sales', gridsize=25)
plt.show()


dplot.std()
dplot.std().plot(kind='bar')
plt.show()
pd.tools.plotting.scatter_matrix(dplot)
plt.show()
pd.plotting.scatter_matrix(dplot)
plt.show()


dplot.Price.plot.kde()
plt.show()
dplot.plot.kde()
plt.show()