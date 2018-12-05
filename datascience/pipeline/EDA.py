from urllib.request import urlopen
import pandas as pd

irisurl = "http://aima.cs.berkeley.edu/data/iris.csv"
iris_open = urllib.request.urlopen(irisurl)
irisdata = pd.read_csv(iris_open, sep=',', decimal='.', header=None,
                        names=['sepal_length', 'sepal_width', 'petal_length',
                               'petal_width', 'target'])
 
