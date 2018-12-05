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
