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

print(np.unique(cardata['Model_year']).sum)

# Relationship plot: continous features
color_palette = 'Blue Green Red Yellow White'.split()
cardata['Origin2'] = cardata['Origin'].map({1: 'USA', 2: 'Europe', 3:'Japan'})
with plt.style.context(('dark_background')):
    g = sns.relplot(x='MPG', y='Horsepower', data=cardata, kind='scatter', 
            hue='Cylinders', col='Origin2', palette=color_palette)
    g.fig.suptitle('')
    g.set_titles('{col_name} Origin')
plt.show()


with plt.style.context(('dark_background')):
    g = sns.relplot(x='MPG', y='Horsepower', data=cardata, kind='scatter', 
            hue='Cylinders', row='Origin2', palette=color_palette)
    g.fig.suptitle('')
    g.set_titles('{col_name} Origin')
plt.show()



# g.set_title() => for titles for AxesSubplot

# category chat
with plt.style.context(('dark_background')):
    g = sns.catplot(x='Cylinders', y='Horsepower', data=cardata, kind='box', 
              palette=color_palette, col='Origin2')
    g.fig.suptitle('Cylinders vs. Horsepower', y=1.03)
    g.set_titles('This is {col_name}')
    g.set(xlabel='Number of Cyclinders')
plt.xticks(rotation=30)
plt.show()
 