import os
import pandas as pd
import numpy as np
from numpy.random import randn
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

# Graph the bins
popbin2.value_counts().plot("bar")
plt.show()


# Clamping dataset
pyramids_data.Male["China"]
china_clamp = pyramids_data.Male.clip(lower=0, upper=1000000)
china_clamp["China"]


# recoding and replacing
vect1 = (randn(30) * 10) .round()
vect1[[1, 2, 5, 6, 18, 20]] = 999
rdata = pd.DataFrame(vect1.reshape(10, 3))
rdata.replace({999: np.nan}, inplace=True)


sleepdata = pd.DataFrame({"Gender": ['m', 'f', 'f', 'f', 'm', 'f'],
                          "Hours": [10, 12, 34, 56, 12, 34],
                          "Weight": [123, 342, 156, 201, 256, 245]})
sleepdata
sleepdata.loc[:, "Gender"].replace({"m": 0, "f": 1}, inplace=True)
sleepdata

sleepdata.mean()

# Derived values
pyramids_data["Total"] = pyramids_data.Male + pyramids_data.Female
pyramids_data.head()

pyramids_data["MalePercentage"] = pyramids_data.Male / pyramids_data.Total
pyramids_data.head()

pyramids_data["FemalePercentage"] = pyramids_data.Female / pyramids_data.Total
pyramids_data.head()

pyramids_data["maleFemaleRatio"] = pyramids_data.Male / pyramids_data.Female
pyramids_data.head()

# Countries with high maleFemaleRatio
pyramids_data.sort_index(inplace=True)
pyramids_data.loc[(slice(None), "Total", 2017),
                  "maleFemaleRatio"].sort_values(ascending=False)


# statistical and mathematical transformation
xbar = pyramids_data.loc[pyramids_data.index.get_level_values(0) != "Total",
                         :].mean()

stdev = pyramids_data.loc[pyramids_data.index.get_level_values(0) != "Total",
                          :].std()

pyramids_data['ScaledCenteredTotal'] = (pyramids_data['Total'] - xbar['Total']) / stdev['Total']

pyramids_data.loc[(slice(None), slice(None), 'Total'), 'ScaledCenteredTotal') = np.nan
pyramids_data.loc[("Afghanistan", 2016)]