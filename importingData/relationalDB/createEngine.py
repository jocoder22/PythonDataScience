#!/usr/bin/env python
# Import necessary module
import os
from sqlalchemy import create_engine, MetaData, Table

path = 'C:/Users/okigboo/Desktop/PythonDataScience/importingData/webData'
os.chdir(path)
# Create engine: engine
engine = create_engine('sqlite:///survey.db')

# Save the table names to a list: table_names
table_names = engine.table_names()

# Print the table names to the shell
print(table_names)   # ['Person', 'Site', 'Survey', 'Visited']


metadata = MetaData()
person = Table('Person', metadata, autoload=True, autoload_with=engine)

# Print the column names
print(person.columns.keys())   # ['id', 'personal', 'family']

# Print full table metadata
print(repr(metadata.tables['Person']))  # same as below

print(repr(person))
# Table('Person', MetaData(bind=None), Column('id', TEXT(), table=<Person>), 
#                                      Column('personal', TEXT(), table=<Person>), 
#                                      Column('family', TEXT(), table=<Person>), 
#                                      schema=None)



