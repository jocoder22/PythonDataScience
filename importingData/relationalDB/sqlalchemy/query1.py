#!/usr/bin/env python
# Import necessary module
import os
from sqlalchemy import create_engine, MetaData, Table, select,  and_ 

path = 'C:/Users/okigboo/Desktop/PythonDataScience/importingData/webData'
os.chdir(path)
# Create engine: engine
engine = create_engine('sqlite:///survey.db')
connection = engine.connect()

metadata = MetaData()

survey = Table('Survey', metadata, autoload=True, autoload_with=engine)

ssmt = select([survey])


# Add a where clause to filter the results to only those for lake
ssmt = ssmt.where(survey.columns.person == 'lake')

# Execute the query to retrieve all the data returned: results
results = connection.execute(ssmt).fetchall()


# Loop over the results and print the taken, quant, and reading
for result in results:
    print(result.taken, result.quant, result.reading)



# persons = ['dyer', 'roe']
# # Append a where clause to match all the states in_ the list states
# ssmt = ssmt.where(survey.columns.person.in_(persons))

# # Loop over the ResultProxy and print the quant and reading
# for result in connection.execute(stmt):
#     print(result.quant, result.reading)


# Append a where clause to select only lake records and rad using and_
ssmt = ssmt.where(
    # The state of California with quantity 'temp'
    and_(survey.columns.person == 'lake',
         survey.columns.quant == 'rad')
        )

# Loop over the ResultProxy printing the taken and reading
for result in connection.execute(ssmt):
    print(result.taken, result.reading)
