import os
import numpy as np
import pandas as pd


os.chdir("C:/Users/Jose/Documents/PythonDataScience1/Code/Code/Section 3")
irisdata = pd.read_csv("iris.csv")

# Explore dataset
irisdata.head()
irisdata.tail()
irisdata.shape
irisdata.columns