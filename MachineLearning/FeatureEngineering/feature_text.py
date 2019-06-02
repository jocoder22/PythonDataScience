#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

sp = '\n\n'
path = r'C:\Users\Jose\Desktop\PythonDataScience\tweeter'
os.chdir(path)


data2 = pd.read_csv('nyt.csv')
data = data2[:50]

# remove unusual characters
data['News_content'] = data['News_content'].str.replace('\r\n', '')
print(data.shape, data.head(), sep=sp, end=sp)


# remove not alphabetic character
data['News_content'] = data['News_content'].str.replace('[^a-zA-Z]', ' ')
print(data.shape, data.head(), sep=sp, end=sp)


# turn textst to lower case
data['News_content'] = data['News_content'].str.lower()
print(data.shape, data.head(), sep=sp, end=sp)

# Calculate the length of each textline
data['textLenght'] = data['News_content'].str.len()
print(data.shape, data.head(), sep=sp, end=sp)


# Number of words in each textline
data['wordNumber'] = data['News_content'].str.split().str.len()
print(data.shape, data.head(), sep=sp, end=sp)


# Find the average length of word
data['average_length'] = data['textLenght'] / data['wordNumber']
print(data.shape, data.head(), sep=sp, end=sp)

# Save data to compressed csv
path = r'C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\FeatureEngineering'
os.chdir(path)
data.to_csv('textdata.csv', index=False, compression='gzip')





