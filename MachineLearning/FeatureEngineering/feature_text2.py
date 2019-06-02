#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r'C:\Users\Jose\Desktop\PythonDataScience\MachineLearning\FeatureEngineering'
os.chdir(path)
data = pd.read_csv('textdata.csv', compression='gzip')

def p(*args):
    for arg in args:
        print(arg, end='\n\n')

p(data.head(),data.shape, data.columns)


