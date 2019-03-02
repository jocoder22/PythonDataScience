#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sp = '\n\n'
path = 'C:\\Users\\Jose\\Desktop\\PythonDataScience\\Projects\\datacamps\\crypto\\'
os.chdir(path)
data = pd.read_csv('crypto.csv')

print(data.head(), data.shape, data.columns, data.info(), sep=sp)
