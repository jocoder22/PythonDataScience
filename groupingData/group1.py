import os
import pandas as pd
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt


# https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/


def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  
sp = {"sep":"\n\n", "end":"\n\n"} 

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


# creating groups
agegroup = pp16long.groupby("Age")
agegroup

countrygroup = pp16long.groupby("Country")
sexgroup = pp16long.groupby("Sex")
agesexgroup = pp16long.groupby(["Age", "Country"])
agegroup
agesexgroup

# Look at the groups
sexgroup.groups
agegroup.groups

# show populaiton based on gender
sexgroup.sum()

# show coutry population, descending order
countrygroup.sum().sort_values("Population", ascending=False)

# show total population based on agegroup
agegroup.sum()
agesexgroup.sum()

# create age categories
agecats = pd.Categorical(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29',
                          '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
                          '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', 
                          '90-94', '95-99', '100+'])

agegroup.sum().loc[agecats]
agegroup.sum().loc[agecats, "Population"].plot("bar")
plt.show()

# Group using index
yeargroup = pyramids_data.groupby(level="Year")
yeargroup.sum()


# get Yearly population
yeargroup.sum().sum(axis=1)
yeargroup.sum().sum(axis=1).plot()
plt.plot(yeargroup.sum().sum(axis=1))
plt.show()

yearcountrygroup = pyramids_data.groupby(level=["Year", "Country"])
yearcountrygroup.sum()
yearcountrygroup.sum().sum(axis=1)

# Yearly populaiton of USA
yearcountrygroup.sum().sum(axis=1).loc[:, "UnitedStates"]

# iterate over sexgroup
for name, data in sexgroup:
    print(name)
    print(data.head())


