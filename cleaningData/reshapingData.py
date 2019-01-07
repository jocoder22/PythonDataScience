#!/usr/bin/env python
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('dob_job_application_filings_subset.csv')
df_subset = df[:, 2:]
airquality = pd.read_csv('airquality.csv')


##### melt, turns data to long and collapsing the columns
pd.melt(frame=df, id_vars='name', value_vars=['treatment a', 'treatment b'],
        var_name='treatment', value_name='result')

# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(frame=airquality, id_vars=['Month','Day'])

# Print the head of airquality_melt
print(airquality_melt.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', 
                          value_name='reading')

# Print the head of airquality_melt
print(airquality_melt.head())





# Melt tb: tb_melt
tb_melt = pd.melt(frame=tb, id_vars=['country','year'])

# Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())




# info:
# the type_country variable in ebola.csv has values string values join by _
# e.g case_chad, death_chad etc

ebola = pd.read_csv('ebola.csv')
# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')

# Create the 'str_split' column, this forms a list 
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

# Create the 'type' column
ebola_melt['type'] = ebola_melt.str_split.str.get(0)

# Create the 'country' column
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())



# Pivot airquality_melt:  create airquality_pivot
airquality_pivot = pd.pivot_table(data=airquality_melt, index=['Month','Day'], 
                            columns='measurement', values='reading')

# Print the head of airquality_pivot
print(airquality_pivot.head())



#### resetting the Index
# Print the index of airquality_pivot
print(airquality_pivot.index)

# Reset the index of airquality_pivot: airquality_pivot_reset
# reset creates a zero base numeric Index
airquality_pivot_reset = airquality_pivot.reset_index()

# Print the new index of airquality_pivot_reset
print(airquality_pivot_reset.index)

# Print the head of airquality_pivot_reset
print(airquality_pivot_reset.head())
