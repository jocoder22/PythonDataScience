#!/usr/bin/env python
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import sklearn
from datetime import datetime, date
import tensorflow as tf

# import talib as tb
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, Embedding
from tensorflow.python.keras.layers import LSTM, Dropout, Flatten
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


import pandas_datareader as pdr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



sp = {"sep":"\n\n", "end":"\n\n"}

symbol = 'AAL'

# symbol = 'RELIANCE.NS'
starttime = datetime(1996, 1, 1)
endtime = date.today()

ALL = pdr.get_data_yahoo(symbol, starttime, endtime)[['Close']]
ALL['maxClose'] = ALL.Close.max()
ALL['minClose'] = ALL.Close.min()
ALL['%Change'] = ALL.Close.pct_change()
print(ALL.head())

# import pip
# sorted(["%s==%s" % (i.key, i.version) for i in pip.get_installed_distributions()])

# https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/


# import pkg_resources
# installed_packages = pkg_resources.working_set
# in_list = sorted(["%s==%s" % (i.key, i.version)
#      for i in installed_packages])
# print(in_list, len(in_list))


# df = pd.read_csv("D:\PythonDataScience\listt.txt", delim_whitespace=True, 
#                  skiprows=2, names=['Modules', 'Version'])
#                 # can use delimiter=r"\s+"
# df2 = pd.read_csv("D:\PythonDataScience\listt2.txt", delimiter="==", 
#                  names=['Modules', 'Version'])

# mylist = list(df['Modules'])
# print(mylist, df2.head(), **sp)

# help('modules')

import sqlite3
with sqlite3.connect('D:\PythonDataScience\sql\survey.db') as con:
    cursor = con.cursor()
    res = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # print(cursor.fetchall())
    for name in res:
        print(name[0])
        



import MySQLdb
db = MySQLdb.connect(user="my-username",passwd="my-password",host="localhost",db="my-databasename")
cursor = db.cursor()
cursor.execute("SELECT * from my-table-name")
data=cursor.fetchall()
for row in data :
    print (row)
db.close()