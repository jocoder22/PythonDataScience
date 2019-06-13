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
cardata = pd.read_csv('car.csv', compression='gzip')

print(cardata.head())


# Plot scatter plot
sns.scatterplot(x='MPG', y='Horsepower', data=cardata)
plt.xlabel('Miles per Gallon')
plt.show()

# count plot
sns.countplot(x='Origin', data=cardata)
plt.xticks(np.arange(3), 'USA Europe Japan'.split())
plt.show()

print(np.unique(cardata['Model_year']))

# Relationship plot: continous features
color_palette = 'Blue Green Red Yellow White'.split()
size_l = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]
with plt.style.context(('dark_background')):
    sns.relplot(x='MPG', y='Horsepower', data=cardata, kind='scatter', 
            hue='Cylinders', col='Origin', palette=color_palette)
plt.show()