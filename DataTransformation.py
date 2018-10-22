import os
import pandas as pd

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 1")

pyramids_data = pd.read_csv("PopPyramids.csv",
                            index_col=["Country", "Age", "Year"])
