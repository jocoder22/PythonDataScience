import os
import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame

import matplotlib.pyplot as plt

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 1")

pyramids_data = pd.read_csv("PopPyramids.csv",
                            index_col=["Country", "Age", "Year"])

pyramids_data = pyramids_data.loc[:, ['Male Population', 'Female Population']]
pyramids_data.columns = pd.Index(["Male", "Female"])
pyramids_data.head()