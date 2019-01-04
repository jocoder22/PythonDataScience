#!/usr/bin/env python
import os
import numpy as np
import pandas as pd

print(os.getcwd())
os.chdir('c:\\Users\\Jose\\Desktop\\')

print(os.getcwd())

# load dataset
data = pd.read_csv("people.csv")

# data.index = data['name']

# pandas looping is similar to looping in a dict
for index, cdict in data.iterrows():
    print('{:10}: {}'.format(index, cdict['age']))
    data.loc[index, 'NAME'] = cdict['name'].upper()

# forming another column
data['Height_^2'] = data['height'].apply(lambda x: x * x)
print(data)

data['Total'] = data['height'].apply(lambda x: x + x)
print(data)