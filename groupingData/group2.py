import os
import pandas as pd
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

# C:\Users\Jose\Documents\PythonDataScience1\Code\Code\Section 1
# C:\Users\Jose\Documents\PythonDataScience1\Code\Code\Section 1
os.chdir("C:/Users/Jose/Documents/PythonDataScience1/Code/Code/Section 1")
pyramids_data = pd.read_csv("PopPyramids.csv")
pyramids_data = pyramids_data.loc[:, ["Year", "Country", "Age", "Male Population",
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

agegroup16 = data2016.groupby("Age")
countrygroup16 = data2016.groupby("Country")
sexgroup16 = data2016.groupby("Sex")
