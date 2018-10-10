import pandas as pd 
import matplotlib.pyplot as plt

import os
print(os.getcwd())

os.chdir('C:/Users/Jose/Documents/PythonDataScience1/Code/Code/Section 1')
data = pd.read_csv('PopPyramids.csv')
data.head()