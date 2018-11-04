import os
import numpy as np
import pandas as pd
from numpy.random import randn
import matplotlib.pyplot as plt


os.chdir("C:/Users/Jose/Documents/PythonDataScience1/Code/Code/Section 1")
pyramids_data = pd.read_csv("PopPyramids.csv")
pyramids_data = pyramids_data.loc[:, ["Year", "Country", "Age",
                                      "Male Population",
                                      "Female Population"]]

pyramids_data.columns = pd.Index(["Year", "Country", "Age", "Male", "Female"])
pyramids_data = pyramids_data.loc[pyramids_data.Age != "Total"]
pyramids_data = pd.melt(pyramids_data, id_vars=["Year", "Country", "Age"],
                        value_name="Population", var_name="Sex")

pyramids_data.head(20)
pyramids_data.tail(29)


# Select 2016 data
data2016 = pyramids_data.loc[pyramids_data.Year == 2016].drop("Year", axis=1)

yeargroup = pyramids_data.groupby("Year")
cyagroup = pyramids_data.groupby(["Country", "Year", "Age"])

# Form groups
agegroup16 = data2016.groupby("Age")
countrygroup16 = data2016.groupby("Country")
sexgroup16 = data2016.groupby("Sex")


# look at the groups
sexgroup16.groups
agegroup16.groups
countrygroup16.groups


# Group-level Calculations
yeargroup.sum()


# calculate summary statistics
agegroup16.sum()
agegroup16.mean()
agegroup16.std()
agegroup16.describe()


sexgroup16.sum()
sexgroup16.mean()
sexgroup16.describe()


countrygroup16.quantile(0.9)


# Using aggregrage() method or the agg()
countrygroup16.agg(np.sum)
countrygroup16.agg([np.sum, np.mean, np.std])
myfunc = lambda x: np.percentile(x, 75) - np.percentile(x, 8.25)
myfunc(np.array([1, 2, 3, 4, 5, 6, 7, 8]))

sexgroup16.agg(myfunc)
myfunc.__name__


sexgroup16.agg([np.sum, myfunc])
sexgroup16.agg((("Total", np.sum), "IQR", myfunc))
