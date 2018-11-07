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