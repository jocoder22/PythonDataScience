import os
import pandas as pd
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 1")
pyramids_data = pd.read_csv("PopPyramids.csv")
pyramids_data = pyramids_data.loc[:, ["Year", "Country", "Age", "Male Population",
                                   "Female Population"]]
pyramids_data.columns = pd.Index(["Year", "Country", "Age", "Male", "Female"])
pyramids_data = pyramids_data.loc[pyramids_data.Age != "Total"]