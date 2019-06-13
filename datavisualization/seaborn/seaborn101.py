#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')
plt.style.use('seaborn-whitegrid')


path = r"C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\UnsupervisedME"
os.chdir(path)
sp = '\n\n'

# load the dataset, 
data = pd.read_csv('car.csv', compression='gzip')

print(data.head())


# Plot scatter plot
sns.scatterplot(x=data['MPG'], y=data['Horsepower'])
plt.show()

# count plot
sns.countplot(x=data['Origin'])
plt.xticks(np.arange(3), 'USA Europe Asia'.split())
plt.show()