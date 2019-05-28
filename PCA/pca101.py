#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests


sp = '\n\n'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'


colname = '''MPG Cylinders Displacement Horsepower Weight Acceleration 
             Model_year Origin Car_name'''.split()

dataset = pd.read_csv(url, names=colname, na_values=['na','Na', '?'],
                skipinitialspace=True, comment='\t', sep=" ", quotechar='"')

dataset.drop(columns='Car_name', inplace=True)
print(dataset.isna().sum())
dataset.dropna(inplace=True)                  