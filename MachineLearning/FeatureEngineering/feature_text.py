#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sp = '\n\n'
path = r'C:\Users\Jose\Desktop\PythonDataScience\tweeter'
os.chdir(path)

data = pd.read_csv('nyt.csv')

# remove unusual characters
data['News_content'] = data['News_content'].str.replace('\r\n', '')
print(data.shape, data.head(), sep=sp, end=sp)


# remove not alphabetic character
data['News_content'] = data['News_content'].str.replace('[^a-zA-Z]', ' ')
print(data.shape, data.head(), sep=sp, end=sp)


# turn textst to lower case
data['News_content'] = data['News_content'].str.lower()
print(data.shape, data.head(), sep=sp, end=sp)