#!/usr/bin/env python
# Import necessary module

import os
from sqlalchemy import create_engine
import sqlite3
import pandas as pd 

os.chdir('c:/Users/Jose/Desktop/PythonDataScience/importingData/webData/')

# Create engine: engine
engine = create_engine('sqlite:///survey.db')

# Save the table names to a list: table_names
table_names = engine.table_names()

# Print the table names to the shell
print(table_names)


# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Visited")
    df = pd.DataFrame(rs.fetchmany(size=10))
    df.columns = rs.keys()

# Print the length of the DataFrame df
print(len(df))
print(df.head())


# using sqlite
# establish a connection
connection = sqlite3.connect("survey.db")

with connection:
    cursor = connection.cursor()
    cursor.execute("SELECT Site.lat, Site.long FROM Site;")
    results = cursor.fetchall()
    df2 = pd.DataFrame(results)
    print(results)


print(len(df2))
print(df2.head())


# using pandas read_sql_query
dfp = pd.read_sql_query("SELECT * FROM Visited", connection)
print(df.equals(dfp))

# using pandas read_sql_query
dfp = pd.read_sql_query("SELECT * FROM Visited", engine)
print(df.equals(dfp))

# explicit programming
connection = sqlite3.connect("survey.db")
cursor = connection.cursor()
cursor.execute("SELECT Site.lat, Site.long FROM Site;")
results = cursor.fetchall()
for r in results:
    print(r)
cursor.close()
connection.close()



