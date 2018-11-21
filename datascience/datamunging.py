import urllib
import pandas as pd
irisurl = "http://aima.cs.berkeley.edu/data/iris.csv"
reqs = urllib.request.Request(irisurl)
iris_open = urllib.request.urlopen(reqs)
iris_data = pd.read_csv(iris_open, sep=',', decimal='.', header=None,
                        names=['sepal_length', 'sepal_width', 'petal_length',
                               'petal_width', 'target'])
iris_data.head()
