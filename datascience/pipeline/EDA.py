from urllib.request import urlopen
import matplotlib.pyplot as plt
import pandas as pd

irisurl = "http://aima.cs.berkeley.edu/data/iris.csv"
iris_open = urlopen(irisurl)
irisdata = pd.read_csv(iris_open, sep=',', decimal='.', header=None,
                       names=['sepal_length', 'sepal_width', 'petal_length',
                              'petal_width', 'target'])
 
# Explore the dataset
irisdata.describe()

# get a boxplot
boxes = irisdata.boxplot(return_type='axes')
plt.show()

irisdata.quantile([0.1, 0.65, 0.95, 0.99])
irisdata.mean()

# more exploration
mydesc = irisdata.describe()
mydesc.loc['count']
mydesc.loc['mean', 'sepal_width']

irisdata.mean().sepal_width
irisdata.sepal_width.mean()

pd.crosstab(irisdata['petal_length'] > irisdata.petal_length.mean(),
            irisdata['petal_width'] > irisdata.petal_width.mean())

# Exlore relationships
scatterplot = irisdata.plot(kind='scatter',
                            x='petal_width', y='petal_length',
                            s=64, c='blue', edgecolors='white')
plt.show()

# check the distribution
disttr = irisdata.petal_width.plot(kind='hist', alpha=0.5, bins=20)
plt.show()
