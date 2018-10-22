import os
import pandas as pd
import matplotlib.pyplot as plt

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

# Explore the bins
popbin.value_counts()
plt.hist(popbin.value_counts())
plt.show()

popbin.value_counts().plot("bar")
plt.show()

# create reasonable bins
popbin2 = pd.cut(pyramids_data.Male, [0, 1000, 10000, 100000,
                                      1000000, 10000000, 100000000])
popbin2.value_counts()

