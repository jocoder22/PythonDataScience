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
Voters.Education.replace({
        "Less than high school":"No Bachelors"
        "High school":"No Bachelors"
        "Associate":"No Bachelors"
        "Bachelors":"Bachelors or More"  
        "Graduate":"Bachelors or More"}, inplace=True)

Voters.AgeGroup.replace({"[18, 30)":"[18, 50)"
                         "[30, 40)":"[18, 50)"
                         "[40, 50)":"[18, 50)"
                         "[50, 60)":"[18, 50)"
                         "[60, 70)":"[18, 50)"
                         "[70, 80)":"[18, 50)"}, inplace=True)