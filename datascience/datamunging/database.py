import numpy as np
import pandas as pd
import h5py, sqlite3
from urllib.request import urlopen

sp = {'sep':'\n\n', 'end':'\n\n'}

droptable = "DROP TABLE IF EXISTS WeatherData;"
createtable = "CREATE TABLE WeatherData \
                (Date INTEGER, City VARCHAR(40), Season VARCHAR(40), \
                Temperature REAL, Grade INTEGER);"
connection = sqlite3.connect("Test01.db")
connection.execute(droptable)
connection.execute(createtable)
connection.commit()


# get the data
mydata = [(20181109, "New York", "Winter", 23.9, 2),
          (20181112, "Brookly", "Winter", 25.0, 2),
          (20181114, "Boston",  "Winter", 33.9, 3),
          (20181116, "Queens", "Winter", 15.9, 1),
          (20181114, "Newark",  "Winter", 13.9, 1),
          (20181116, "Hicksville", "Winter", 10.9, 1)]
inserttable1 = "INSERT INTO WeatherData VALUES(?, ?, ?, ?, ?)"
connection.executemany(inserttable1, mydata)
connection.commit()

# select query
selectAll = "SELECT * FROM WeatherData;"
workingdata = pd.read_sql_query(selectAll, connection)
connection.close()

workingdata.Date.dtype
workingdata['Date'] = pd.to_datetime(workingdata['Date'].astype(str), format='%Y%m%d')
print(workingdata.head(), workingdata.shape, workingdata.Date.dtype, **sp)


# Using h5py

irisurl = "http://aima.cs.berkeley.edu/data/iris.csv"
irisdata = urlopen(irisurl)
iris_data = pd.read_csv(irisdata, sep=',', decimal='.', header=None,
                        names=['sepal_length', 'sepal_width', 'petal_length',
                               'petal_width', 'target'])

storage = pd.HDFStore("firststorage.h5")
storage['iris'] = irisdata
storage.close()

# Open the file
mystorage = pd.HDFStore("firststorage.h5")
mystorage.keys()

iris_pd = mystorage['iris']

# Basic statistics
myfunction = {'Septal_length': ['mean', 'max'],
              'Septal_width': ['std', 'min'],
              'petal_length': ['mean', 'var'],
              'petal_width': ['max', 'med']}

Summary_stats = iris_data.groupby(['target']).agg(myfunction)
