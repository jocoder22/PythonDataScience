import os
import pandas as pd

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 1")

pyramids_data = pd.read_csv("PopPyramids.csv",
                            index_col=["Country", "Age", "Year"])


# Explore the data
pyramids_data.head()
pyramids_data.columns
pyramids_data.columns.tolist()

# Drop column region
pyramids_data.drop("Region", 1, inplace=True)
pyramids_data.head()

# select two columns and change index
pyramids_data = pyramids_data.loc[:, ['Male Population', 'Female Population']]
pyramids_data.columns = pd.Index(["Male", "Female"])
pyramids_data.head()


# Bining data
popbin = pd.cut(pyramids_data.Male, 10)
popbin