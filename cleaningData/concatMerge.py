import os
import glob
import pandas as pd
from sqlalchemy import create_engine

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  


uber1, uber2, uber3 = pd.read_csv(['uber1.csv', 'uber2.csv', 'uber3.csv'])


# Concatenate uber1, uber2, and uber3 row wise: row_concat
row_concat = pd.concat([uber1, uber2, uber3])

# Print the shape of row_concat
print(row_concat.shape)

# Print the head of row_concat
print(row_concat.head())




ebola_melt, status_country =  pd.read_csv(['ebola_melt.csv', 'status_country.csv'])

# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat([ebola_melt, status_country], axis=1)

# Print the shape of ebola_tidy
print(ebola_tidy.shape)

# Print the head of ebola_tidy
print(ebola_tidy.head())






# using glob the search and concatenate many dataset
# Write the pattern: pattern
pattern = '*.csv'

# Save all file matches: csv_files
csv_files = glob.glob(pattern)

# Print the file names
print(csv_files)

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[1])

# Print the head of csv2
print(csv2.head())




# loading the file matches
# Create an empty list: frames
frames = []

#  Iterate over csv_files
for csv in csv_files:

    #  Read csv into a DataFrame: df
    df = pd.read_csv(csv)
    
    # Append df to frames
    frames.append(df)

# Concatenate frames into a single DataFrame: uber
uber = pd.concat(frames)

# Print the shape of uber
print(uber.shape)






#  Merging dataset, colunm wise
# using the dataset from the database
# change the directory and create the engine
os.chdir('c:/Users/Jose/Desktop/PythonDataScience/importingData/webData/')

# Create engine: engine
engine = create_engine('sqlite:///survey.db')

# reading data from the tables
visited = pd.read_sql_query("SELECT * FROM Visited", engine)
site = pd.read_sql_query("SELECT * FROM Site", engine)

# Merge the DataFrames: o2o
o2o = pd.merge(left=site, right=visited, on=None,left_on='name', right_on='site')

# Print o2o
print(o2o)

