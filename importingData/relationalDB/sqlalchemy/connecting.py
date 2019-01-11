#!/usr/bin/env python
# Import necessary module

import os
from sqlalchemy import create_engine, MetaData, Table, select


path = 'C:/Users/okigboo/Desktop/PythonDataScience/importingData/webData'
os.chdir(path)
# Create engine: engine
engine = create_engine('sqlite:///survey.db')
connection = engine.connect()

metadata = MetaData()

print(engine.table_names())  # ['Person', 'Site', 'Survey', 'Visited']

survey = Table('Visited', metadata, autoload=True, autoload_with=engine)

ssmt = select([survey])

print(ssmt)
results = connection.execute(ssmt).fetchall()

print(results)
print(results[0])
print(results[0].keys())


# Get the first row of the results by using an index: first_row
first_row = results[0]

# Print the first row of the results
print(first_row)

# Print the first column of the first row by using an index
print(first_row[0])

# Print the 'family' column of the first row by using its name
print(first_row['quant'])


# Add a where clause to filter the results to only those for lake
ssmt = ssmt.where(survey.columns.person == 'lake')

# Execute the query to retrieve all the data returned: results
results = connection.execute(ssmt).fetchall()
