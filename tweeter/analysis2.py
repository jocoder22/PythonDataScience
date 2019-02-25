#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import text_preprocessing

sp = '\n\n'
# path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
os.chdir(path)

data = pd.read_csv('nyt_clean.csv')

text_clean = []


for text in data['News_content']:
    text_clean.extend(text_preprocessing.preprocessText(text))

    

print(text_clean[:10])
