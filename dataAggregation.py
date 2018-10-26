import os
import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame

import matplotlib.pyplot as plt

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 1")

pyramids_data = pd.read_csv("PopPyramids.csv",
                            index_col=["Country", "Age", "Year"])