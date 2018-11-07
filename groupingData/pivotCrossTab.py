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

