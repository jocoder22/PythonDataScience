#!/usr/bin/env python
# Import necessary module
import os
from sqlalchemy import create_engine, MetaData, Table, select, desc


path = 'C:/Users/okigboo/Desktop/PythonDataScience/importingData/webData'
os.chdir(path)
# Create engine: engine
engine = create_engine('sqlite:///survey.db')
connection = engine.connect()

metadata = MetaData()

survey = Table('Survey', metadata, autoload=True, autoload_with=engine)

ssmt = select([survey])


############## order_by
# Build a query to select the person column: stmt
stmt = select([survey.columns.person])

# Order stmt by the reading column
stmt = stmt.order_by(survey.columns.reading)

# Execute the query and store the results: results
results = connection.execute(stmt).fetchall()

# Print the first 10 results
print(results[:10])


# Import desc

# Build a query to select the person column: stmt
stmt = select([survey.columns.person])

# Order stmt by reading in descending order: rev_stmt
rev_stmt = stmt.order_by(desc(survey.columns.reading))

# Execute the query and store the results: rev_results
rev_results = connection.execute(rev_stmt).fetchall()

# Print the first 10 rev_results
print(rev_results[:10])


# Build a query to select person and quant: stmt
stmt = select([survey.columns.person, survey.columns.quant])

# Append order by to ascend by person and descend by quant
stmt = stmt.order_by(survey.columns.person, survey.columns.quant)

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print the first 20 results
print(results[:20])


# Build a query to select person and quant: stmt
stmt = select([survey.columns.person, survey.columns.quant])

# Append order by to ascend by state and descend by quant
stmt = stmt.order_by(survey.columns.person, desc(survey.columns.quant))

# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()

# Print the first 20 results
print(results[:20])
