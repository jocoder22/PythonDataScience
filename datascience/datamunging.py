import os
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
print(os.getcwd()) # C:/Users/Jose
csvdataset = pd.read_csv(u'~/Desktop/new2.csv',sep=',' , parse_dates=[0]) 
csvdataset


# fill in missing values
csvdataset.fillna(50)
# Alternatively, using the column mean
csvdataset.fillna(csvdataset.mean(axis=0))

# Handling poorly formatted dataset
baddataset = pd.read_csv(u'~/Desktop/bad.csv',sep=',', error_bad_lines=False) 
# Info : b'Skipping line 4: expected 3 fields, saw 4\n'
baddataset