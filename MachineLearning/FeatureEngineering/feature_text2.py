#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r'C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\FeatureEngineering'
os.chdir(path)
data = pd.read_csv('textdata.csv', compression='gzip')

def xprint(d):
    for arg in (d.head(), d.info(), d.shape, d. columns):
        print(arg, end='\n\n')

xprint(data)


