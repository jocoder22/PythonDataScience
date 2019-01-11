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
site = Table('Site', metadata, autoload=True, autoload_with=engine)
visits = Table('Visited', metadata, autoload=True, autoload_with=engine)
person = Table('Person', metadata, autoload=True, autoload_with=engine)

ssmt = select([survey])

############# join predefined join table
# Build a statement to join survey and state_fact tables: stmt
stmt = select([survey.columns.pop2000, state_fact.columns.abbreviation])

# Execute the statement and get the first result: result
result = connection.execute(stmt).first()

# Loop over the keys in the result object and print the key and value
for key in result.keys():
    print(key, getattr(result, key))


# Build a statement to select the survey and state_fact tables: stmt
stmt = select([survey, state_fact])

# Add a select_from clause that wraps a join for the survey and state_fact
# tables where the survey state column and state_fact name column match
stmt = stmt.select_from(
    survey.join(state_fact, survey.columns.state == state_fact.columns.name))

# Execute the statement and get the first result: result
result = connection.execute(stmt).first()

# Loop over the keys in the result object and print the key and value
for key in result.keys():
    print(key, getattr(result, key)


          # Start a while loop checking for more results
          while more_results:
          # Fetch the first 50 results from the ResultProxy: partial_results
          partial_results=results_proxy.fetchmany(50)

          # if empty list, set more_results to False
          if partial_results == []:
          more_results=False

          # Loop over the fetched records and increment the count for the state
          for row in partial_results:
          if row.state in state_count:
              state_count[row.state] += 1
          else:
              state_count[row.state]=1

          # Close the ResultProxy, and thus the connection
          results_proxy.close()

          # Print the count by state
          print(state_count)
