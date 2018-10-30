import os
import pandas as pd
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 1")
pyramids_data = pd.read_csv("PopPyramids.csv",
                            index_col=["Country", "Year", "Age"])


# select two columns and change index
pyramids_data = pyramids_data.loc[:, ['Male Population', 'Female Population']].drop("Total",
                                    axis=0, level="Age")
pyramids_data.columns = pd.Index(["Male", "Female"])
pyramids_data.sort_index(inplace=True)
pyramids_data.head()


# select on 2016 data
pyramids_data16 = pyramids_data.loc[(slice(None), 2016), :]
pyramids_data16.index = pyramids_data16.index.droplevel("Year")
pyramids_data16.head()


# store data in columns
ppcolumns = pyramids_data16.reset_index()
ppcolumns.head()


# go long form
pp16long = pd.melt(ppcolumns, id_vars=["Country", "Age"],
                              value_name="Population",
                              var_name="Sex")
pp16long.head()
