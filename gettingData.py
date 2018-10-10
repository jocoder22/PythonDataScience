import pandas as pd 
import matplotlib.pyplot as plt

import os
print(os.getcwd())

filename1 = "C:/Users/Jose/Documents/PythonDataScience1/Code/Code/Section 1/PopPyramids.csv"
data = pd.read_csv(filename1)
data.head()