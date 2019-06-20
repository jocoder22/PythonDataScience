#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode 
# plt.style.use('ggplot')
# plt.style.use('seaborn-whitegrid')


path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = '\n\n'

# load the dataset, 
car = pd.read_csv('car.csv', compression='gzip')
car4 = car.copy()

print(car.head(), end=sp)
print(pd.unique(car['Model_year']), end=sp)

modelyear = car['Model_year'].value_counts()
cyc = car['Cylinders'].value_counts()

cyc.plot(kind='bar')
plt.show()


sns.set()
# bar plot
cyc.plot(kind='bar')
plt.show()

# distplot 
sns.distplot(car['MPG'])
plt.show()

# # pairplot
# car2 = car.iloc[:, :4]
# sns.pairplot(car2)
# plt.show()


# lmplot
sns.lmplot(x='Displacement', y='MPG', data=car)
plt.show()


# jointplot
sns.jointplot(x='Displacement', y='MPG', data=car)
plt.show()



car.loc[car['Model_year'] >= 70, 'Year'] = 'Early 70s'
car.loc[car['Model_year'] >= 75, 'Year'] = 'Late 70s'
car.loc[car['Model_year'] >= 80, 'Year'] = 'Early 80s'

# # boxplot
# gg = sns.boxplot(x='Year', y='MPG', data=car)
# plt.show()


# # swarmplot
# gs = sns.swarmplot(x='Year', y='MPG', data=car)
# gs.set_xticklabels(gs.get_xticklabels(), rotation=45)
# plt.show()

# # heatmap
# df = car.pivot_table(index='Cylinders', columns='Displacement', values='MPG', aggfunc='mean')
# print(df)
# sns.heatmap(df)
# plt.show()


# heatmap 2
df2 = car.pivot_table('Cylinders', 'Year', 'Origin', 'median')
sns.heatmap(df2, annot=True,  linewidths=.5, 
        cmap="YlGnBu", xticklabels='USA Europe Japan'.split())
# plt.xticks(np.arange(3), 'USA Europe Japan'.split())
plt.show()
