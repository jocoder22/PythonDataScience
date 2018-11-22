import urllib
import pandas as pd
irisurl = "http://aima.cs.berkeley.edu/data/iris.csv"
reqs = urllib.request.Request(irisurl)
iris_open = urllib.request.urlopen(reqs)
iris_data = pd.read_csv(iris_open, sep=',', decimal='.', header=None,
                        names=['sepal_length', 'sepal_width', 'petal_length',
                               'petal_width', 'target'])


# Explore the dataset
iris_data.head()
iris_data.tail()
iris_data.columns

Y_series = iris_data['target']
X_frame = iris_data[['sepal_length', 'sepal_width']]


Y_series.shape
X_frame.shape

# Reading csv files
csvdataset = pd.read_csv(u'C:/Users/Jose/Desktop/new2.csv',sep=',') 
csvdataset