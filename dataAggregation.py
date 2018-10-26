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
pyramids_data.sort(inplace=True)
pyramids_data.head()

# select on 2016 data
pyramids_data_2016 = pyramids_data.loc[(slice(None), 2016), :]
pyramids_data_2016 = pyramids_data_2016.index.droplevel("Year")
pyramids_data_2016.head()

# remove the index, turns indexes to columns
pyramids_dataColumns = pyramids_data_2016.reset.index()
pyramids_dataColumns.head()