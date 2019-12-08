#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from zipfile import ZipFile

# Create file path: file_path
# file_path = 'Summer Olympic medallists 1896 to 2008 - EDITIONS.tsv'

def print2(**args):
    for arg in args:
        print(arg, end='\n\n', sep='\n\n')

url = 'https://assets.datacamp.com/production/repositories/516/datasets/2d14df8d3c6a1773358fa000f203282c2e1107d6/Summer%20Olympic%20medals.zip'

response = requests.get(url)

# unzip the content

zipp = ZipFile(BytesIO(response.content))
print(zipp.namelist())

mylist = [filename for filename in zipp.namelist()]
print2(mylist)
mymedal2 = pd.read_csv(zipp.open(mylist[8]), sep='\t')
# mymedal2 = pd.read_csv(zipp.open(file_path), sep='\t')

# Load DataFrame from file_path: editions
editions = pd.read_csv(zipp.open(mylist[8]), sep='\t')
# editions = pd.read_csv(zipp.open(file_path), sep='\t')

ioc_codes = pd.read_csv(zipp.open(mylist[9]))
allmedals = pd.read_csv(zipp.open(mylist[7]), sep='\t', skiprows=4)

# Extract the relevant columns: editions
# editions = editions.loc[:, ['Edition', 'Grand Total', 'City', 'Country']] -- data with index
editions = editions[['Edition', 'Grand Total',
                     'City', 'Country']]  # data without index
# Print editions DataFrame
print(editions)

newlist = allmedals[['Athlete', 'NOC', 'Medal','Edition']]

# # Create empty dictionary: medals_dict
# medals_dict = {}

# for year in editions['Edition']:

#     # Create the file path: file_path
#     file_path = 'summer_{:d}.csv'.format(year)

#     # Load file_path into a DataFrame: medals_dict[year]
#     medals_dict[year] = pd.read_csv(file_path)  # this created  the year index

#     # Extract relevant columns: medals_dict[year]
#     medals_dict[year] = medals_dict[year][['Athlete', 'NOC', 'Medal']]

#     # Assign year to column 'Edition' of medals_dict
#     medals_dict[year]['Edition'] = year

# # Concatenate medals_dict: medals
# medals = pd.concat(medals_dict)
# # medals = pd.concat(medals_dict, ignore_index=True) ## this ignores the year index

# # Print first and last 5 rows of medals
# print(medals.head())
# print(medals.tail())


# Construct the pivot_table: medal_counts
# medal_counts = medals.pivot_table(
#     index='Edition', columns='NOC', values='Athlete', aggfunc='count')


medal_counts = newlist.pivot_table(
    index='Edition', columns='NOC', values='Athlete', aggfunc='count')
# Print the first & last 5 rows of medal_counts
print(medal_counts.head())
print(medal_counts.tail())


# Set Index of editions: totals
totals = editions.set_index('Edition')

# Reassign totals['Grand Total']: totals
totals = totals['Grand Total']

# Divide medal_counts by totals: fractions
fractions = medal_counts.divide(totals, axis='rows')

# Print first & last 5 rows of fractions
print(fractions.head())
print(fractions.tail())


# Apply the expanding mean: mean_fractions
mean_fractions = fractions.expanding().mean()

# Compute the percentage change: fractions_change
fractions_change = mean_fractions.pct_change() * 100

# Reset the index of fractions_change: fractions_change
fractions_change = fractions_change.reset_index()

# Print first & last 5 rows of fractions_change
print(fractions_change.head())
print(fractions_change.tail())


# Left join editions and ioc_codes: hosts
hosts = pd.merge(editions, ioc_codes, how='left')

# Extract relevant columns and set index: hosts
hosts = hosts[['Edition', 'NOC']].set_index('Edition')

# Fix missing 'NOC' values of hosts
print(hosts.loc[hosts.NOC.isnull()])
hosts.loc[1972, 'NOC'] = 'FRG'
hosts.loc[1980, 'NOC'] = 'URS'
hosts.loc[1988, 'NOC'] = 'KOR'

# Reset Index of hosts: hosts
hosts = hosts.reset_index()

# Print hosts
print(hosts)


# Reshape fractions_change: reshaped
reshaped = pd.melt(fractions_change, id_vars='Edition', value_name='Change', )

# Print reshaped.shape and fractions_change.shape
print(reshaped.shape, fractions_change.shape)

# Extract rows from reshaped where 'NOC' == 'CHN': chn
chn = reshaped[reshaped['NOC'] == 'CHN']

# Print last 5 rows of chn with .tail()
print(chn.tail())


# Import pandas

# Merge reshaped and hosts: merged
merged = pd.merge(reshaped, hosts)

# Print first 5 rows of merged
print(merged.head())

# Set Index of merged and sort it: influence
influence = merged.set_index('Edition').sort_index()

# Print first 5 rows of influence
print(influence.head())


# Import pyplot


# Extract influence['Change']: change
change = influence['Change']

# Make bar plot of change: ax
ax = change.plot(kind='bar')

# Customize the plot to improve readability
ax.set_ylabel("% Change of Host Country Medal Count")
ax.set_title("Is there a Host Country Advantage?")
ax.set_xticklabels(editions['City'])

# Display the plot
plt.show()


# Build a statement to select the state, sum of 2008 population and census
# division name: stmt
stmt = select([
    census.columns.state,
    func.sum(census.columns.pop2008),
    state_fact.columns.census_division_name
])

# Append select_from to join the census and state_fact tables by the census state and state_fact name columns
stmt = stmt.select_from(
    census.join(state_fact, census.columns.state == state_fact.columns.name)
)

# Append a group by for the state_fact name column
stmt = stmt.group_by(state_fact.columns.name)

# Execute the statement and get the results: results
results = connection.execute(stmt).fetchall()

# Loop over the the results object and print each record.
for record in results:
    print(record)


# Make an alias of the employees table: managers
managers = employees.alias()

# Build a query to select manager's and their employees names: stmt
stmt = select(
    [managers.columns.name.label('manager'),
     employees.columns.name.label('employee')]
)

# Match managers id with employees mgr: stmt
stmt = stmt.where(managers.columns.id == employees.columns.mgr)

# Order the statement by the managers name: stmt
stmt = stmt.order_by(managers.columns.name)

# Execute statement: results
results = connection.execute(stmt).fetchall()

# Print records
for record in results:
    print(record)


# Make an alias of the employees table: managers
managers = employees.alias()

# Build a query to select managers and counts of their employees: stmt
stmt = select([managers.columns.name, func.count(employees.columns.id)])

# Append a where clause that ensures the manager id and employee mgr are equal
stmt = stmt.where(managers.columns.id == employees.columns.mgr)

# Group by Managers Name
stmt = stmt.group_by(managers.columns.name)

# Execute statement: results
results = connection.execute(stmt).fetchall()

# print manager
for record in results:
    print(record)
