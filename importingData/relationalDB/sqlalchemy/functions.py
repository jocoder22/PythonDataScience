#!/usr/bin/env python
# Import necessary module
#!/usr/bin/env python
# Import necessary module

import os
from sqlalchemy import create_engine, MetaData, Table, select,  func, desc, case, cast
from sqlalchemy import Table, Column, String, Integer, Float, Boolean
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:/Users/okigboo/Desktop/PythonDataScience/importingData/webData'
os.chdir(path)
# Create engine: engine
engine = create_engine('sqlite:///survey.db')
connection = engine.connect()

metadata = MetaData()

survey = Table('Survey', metadata, autoload=True, autoload_with=engine)

ssmt = select([survey])


###### func
# Build a query to count the distinct persons values: stmt
stmt = select([func.count(survey.columns.person.distinct())])

# Execute the query and store the scalar result: distinct_person_count
distinct_person_count = connection.execute(stmt).scalar()

# Print the distinct_person_count
print(distinct_person_count)


# Build a query to select the person and count of readings by person: stmt
stmt = select([survey.columns.person, func.count(survey.columns.reading)])

# Group stmt by person
stmt = stmt.group_by(survey.columns.person)

# Execute the personment and store all the records: results
results = connection.execute(stmt).fetchall()

# Print results
print(results)

# Print the keys/column names of the results returned
print(results[0].keys())


# Build an expression to calculate the sum of taken labeled as TotalTaken
taken_sum = func.sum(survey.columns.taken).label('TotalTaken')

# Build a query to select the person and sum of taken: stmt
stmt = select([survey.columns.person, taken_sum])

# Group stmt by person
stmt = stmt.group_by(survey.columns.person)

# Execute the personment and store all the records: results
results = connection.execute(stmt).fetchall()

# Print results
print(results)

# Print the keys/column names of the results returned
print(results[0].keys())


########### with pandas

# Create a DataFrame from the results: df
df = pd.DataFrame(results)

# Set column names
df.columns = results[0].keys()

# Print the Dataframe
print(df)


# Create a DataFrame from the results: df
df = pd.DataFrame(results)

# Set Column names
df.columns = results[0].keys()
label = list(df.columns)
objects = tuple(df.person)
y_pos = np.arange(len(objects))
# Print the DataFrame
print(df)

# Plot the DataFrame
df.plot.bar()
plt.xlabel(label[0])
plt.ylabel(label[1])

plt.xticks(y_pos, objects)

plt.show()


# Build query to return person names by population difference from 2008 to 2000: stmt
stmt = select([survey.columns.person, (survey.columns.taken -
                                      survey.columns.reading).label('pop_change')])

# Append group by for the person: stmt
stmt = stmt.group_by(survey.columns.person)

# Append order by for pop_change descendingly: stmt
stmt = stmt.order_by(desc('pop_change'))

# Return only 5 results: stmt
stmt = stmt.limit(5)

# Use connection to execute the personment and fetch all results
results = connection.execute(stmt).fetchall()

# Print the person and population change for each record
for result in results:
    print('{}:{}'.format(result.person, result.pop_change))


# import case, cast and Float from sqlalchemy

# Build an expression to calculate dyer population in 2000
dyer_reading = func.sum(
    case([
        (survey.columns.person == 'dyer', survey.columns.reading)
    ], else_=0))

# Cast an expression to calculate total population in 2000 to Float
total_reading = cast(func.sum(survey.columns.reading), Float)

# Build a query to calculate the percentreading of dyers in 2000: stmt
stmt = select([dyer_reading / total_reading * 100])

# Execute the query and store the scalar result: percent_dyer
percent_dyer = connection.execute(stmt).scalar()

# Print the percentreading
print(percent_dyer)
