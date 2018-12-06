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
disttr20 = irisdata.petal_width.plot(kind='hist', alpha=0.5, bins=20)
plt.show()

disttr12 = irisdata.petal_width.plot(kind='hist', alpha=0.5, bins=12)
plt.show()



img = plt.matshow(cov_data, cmap=plt.cm.rainbow)
plt.colorbar(img, ticks=[-1, 0, 1], fraction=0.045)
for x in range(cov_data.shape[0]):
    for y in range(cov_data.shape[1]):
        plt.text(x, y, "%0.2f" % cov_data[x,y], 
                    size=12, color='black', ha="center", va="center")
plt.show()
