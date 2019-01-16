#!/usr/bin/env python
# import os
# import numpy as np
import pandas as pd

# print(os.getcwd())
# os.chdir('c:\\Users\\Jose\\Desktop\\')

# print(os.getcwd())

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


data['BMI'] =  data['weight'] / (data['height'] * 0.3048)  ** 2
# data['BMI'] = data['BMI'].apply(lambda x: '%.2f' % x)
data['BMI'] = data['BMI'].apply(lambda x: f'{x:.2f}')
print(data)



newdict = {}
# Iterate over the columns in DataFrame
for entry in data:
    if entry in newdict.keys():
        newdict[entry] += 1
    else:
        newdict[entry] = 1
    print(entry)
# Print the populated dictionary
print(newdict)





# Initialize an empty dictionary: counts
counts = {}
# Iterate over the file chunk by chunk
for chunk in pd.read_csv('people.csv', chunksize=1):

    # Iterate over the column 'name' in  the DataFrame
    for entry in chunk['name']:
        if entry in counts.keys():
            counts[entry] += 1
        else:
            counts[entry] = 1

# Print the populated dictionary
print(counts)


SELECT country_code, size,
CASE WHEN size > 50000000 THEN 'large'
WHEN size > 1000000 THEN 'medium'
ELSE 'small' END
AS popsize_group
INTO pop_plus -- creates a new table to store the query result
FROM populations
WHERE year = 2015;

