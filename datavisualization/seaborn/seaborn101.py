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

print(cardata.head(), end=sp)
print(pd.unique(cardata['Model_year']), end=sp)
print(cardata['Model_year'].value_counts(), end=sp)

cardata.loc[cardata['Model_year'] >= 70, 'Year'] = 'Early 70s'
cardata.loc[cardata['Model_year'] >= 75, 'Year'] = 'Late 70s'
cardata.loc[cardata['Model_year'] >= 80, 'Year'] = 'Early 80s'
print(cardata['Year'].value_counts(), end=sp)

# numpy.where => replace where where condition is true, and where condition if false
# numpy.where is a binary replacement method
cardata['Year2'] = np.where(cardata['Year'].isin(['Early 70s', 'Early 80s']),'Olddata', 'NewData')
print(cardata['Year2'].value_counts(), end=sp)

# pandas.where => Replace values where the condition is False.
cardata['Year3'] = cardata['Year'].where(lambda x: x.isin(['Early 70s', 'Early 80s']), 'NewData')
print(cardata['Year3'].value_counts(), end=sp)
'''

# Plot scatter plot
sns.scatterplot(x='MPG', y='Horsepower', data=cardata)
plt.xlabel('Miles per Gallon')
plt.show()

# count plot
sns.countplot(x='Origin', data=cardata)
plt.xticks(np.arange(3), 'USA Europe Japan'.split())
plt.show()


# Relationship plot: continous features
color_palette = 'Blue Green Red Yellow White'.split()
cardata['Origin2'] = cardata['Origin'].map({1: 'USA', 2: 'Europe', 3:'Japan'})

# arrange plots in columns
with plt.style.context(('dark_background')):
    g = sns.relplot(x='MPG', y='Horsepower', data=cardata, kind='scatter', 
            hue='Cylinders', col='Origin2', palette=color_palette)
    g.fig.suptitle('')
    g.set_titles('{col_name} Origin')
plt.show()

# arrange plots in rows
with plt.style.context(('dark_background')):
    g = sns.relplot(x='MPG', y='Horsepower', data=cardata, kind='scatter', 
            hue='Cylinders', row='Origin2', palette=color_palette)
    g.fig.suptitle('')
    g.set_titles('{row_name} Origin')
plt.show()


with plt.style.context(('dark_background')):
    g = sns.relplot(x='MPG', y='Horsepower', data=cardata, kind='scatter', 
            col='Cylinders', row='Origin2', palette=color_palette)
    g.fig.suptitle('')
    g.set_titles('{row_name} {col_name} Cylinders', y=0.000000001)
plt.show()

# g.set_title() => for titles for AxesSubplot

# category chat
with plt.style.context(('dark_background')):
    g = sns.catplot(x='Cylinders', y='Horsepower', data=cardata, kind='box', 
              palette=color_palette, col='Origin2')
    g.fig.suptitle('Cylinders vs. Horsepower', y=1.03)
    g.set_titles('This is {col_name}', y=0.0)
    g.set(xlabel='Number of Cyclinders')
    plt.xticks(rotation=30)
plt.show()
 

 '''