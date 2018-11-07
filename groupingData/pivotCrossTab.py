import os
import numpy as np
import pandas as pd


os.chdir("C:/Users/Jose/Documents/PythonDataScience1/Code/Code/Section 3")
Voters = pd.read_csv("VoterData.csv")

# explore dataset
Voters.shape
Voters.columns
Voters.head()
Voters.tail()


# post processing
Voters.Education.replace({"Less than high school":"No Bachelors",
                          "High school":"No Bachelors",
                          "Associate":"No Bachelors",
                          "Bachelors":"Bachelors or More",
                          "Graduate":"Bachelors or More"}, inplace=True)

Voters.AgeGroup.replace({"[18, 30)":"[18, 50)",
                         "[30, 40)":"[18, 50)",
                         "[40, 50)":"[18, 50)",
                         "[50, 60)":"[50, 90)",
                         "[60, 70)":"[50, 90)",
                         "[70, 80)":"[50, 90)",
                         "[80, 90)":"[50, 90)"}, inplace=True)

# Categorize dataset
Voters.Education = pd.Categorical(Voters.Education)
Voters.AgeGroup = pd.Categorical(Voters.AgeGroup)
Voters.VotedFor = pd.Categorical(Voters.VotedFor)

# More exploration
Voters.dtypes
Voters.head()

# Read in second dataset
pyramids_data = pd.read_csv("PopPyramids.csv")
UspopData = pyramids_data.loc[(pyramids_data.Country == "UnitedStates") & 
                              (pyramids_data.Age != "Total"), 
                              ["Year", "Age", "Male Population", "Female Population"]]
UspopData.columns = pd.Index(["Year", "Age", "Male", "Female"]) 

# explore dataset
UspopData.head()
UspopData.shape
UspopData.dtypes

# turn to long-format
Uspoplong = pd.melt(UspopData, id_vars=["Year", "Age"],
                               var_name="Sex",
                               value_name="Population")

Uspoplong.head()
Uspoplong.shape
Uspoplong.dtypes
Uspoplong.tail()

# CrossTabulation
pd.crosstab(Voters.Education, Voters.VotedFor)
pd.crosstab(Voters.Education, Voters.VotedFor, margins=True)
pd.crosstab(Voters.AgeGroup, Voters.Education)
pd.crosstab(Voters.AgeGroup, Voters.VotedFor, margins=True)

pd.crosstab([Voters.AgeGroup, Voters.Education], Voters.Registered)
pd.crosstab([Voters.AgeGroup, Voters.Education], Voters.Registered, margins=True)
