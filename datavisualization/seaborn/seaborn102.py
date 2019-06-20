#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')
# plt.style.use('seaborn-whitegrid')


path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = '\n\n'

# load the dataset, 
car = pd.read_csv('car.csv', compression='gzip')

print(car.head(), end=sp)
print(pd.unique(car['Model_year']), end=sp)

modelyear = car['Model_year'].value_counts()
cyc = car['Cylinders'].value_counts()

cyc.plot(kind='bar')
plt.show()


sns.set()
cyc.plot(kind='bar')
plt.show()