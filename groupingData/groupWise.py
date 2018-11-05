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

# replace missing
fillingdata = np.array([False] * (150 * 5))
fillingdata[np.random.choice(np.array(150 * 5), size=150, replace=False)] = True
fillingdata = fillingdata.reshape(150, 5)
fillingdata[:, 4] = False
fillingdata[:10, :]