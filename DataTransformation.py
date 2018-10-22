import os
import pandas as pd

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 1")

pyramids_data = pd.read_csv("PopPyramids.csv",
                            index_col=["Country", "Age", "Year"])


# Explore the data
pyramids_data.head()
pyramids_data.columns
pyramids_data.columns.tolist()
