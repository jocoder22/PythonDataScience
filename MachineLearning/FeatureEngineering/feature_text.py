#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sp = '\n\n'
path = r'C:\Users\Jose\Desktop\PythonDataScience\tweeter'
os.chdir(path)

data = pd.read_csv('nyt.csv')


data['News_content'] = data['News_content'].str.replace('\r\n', '')
print(data.shape, data.head(), sep=sp, end=sp)