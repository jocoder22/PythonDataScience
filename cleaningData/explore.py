#!/usr/bin/env python
# Import pandas
import re 
import pandas as pd
import matplotlib.pyplot as plt

# Read the file into a DataFrame: df
df = pd.read_csv('dob_job_application_filings_subset.csv')

# Print the head of df
print(df.head())

# Print the tail of df
print(df.tail())

# Print the shape of df
print(df.shape)

# Print the columns of df
print(df.columns)

print(df.info())

# find the dtype
print(df.dtypes)

df_subset = df[:, 2:]
# Print the head and tail of df_subset
print(df_subset.head())
print(df_subset.tail())


df.info()
df.describe()
df['Existing Zoning Sqft'].describe()


# check for missing values
# dropna=False will count the missing values
df.continent.value_counts(dropna=False)
df['continent'].value_counts(dropna=False)

df.country.value_counts(dropna=False).head()
df.fertility.value_counts(dropna=False).head()
df.populaton.value_counts(dropna=False).head()


# summary statistics for numeric variable
df.describe()



# explore using regular expression
# Create a Series called countries consisting of the 'country' column of gapminder.
# Drop all duplicates from countries using the .drop_duplicates() method.
# Write a regular expression that tests your assumptions of what characters belong in countries:
# Anchor the pattern to match exactly what you want by placing a ^ in the beginning 
# and $ in the end.
# Use A-Za-z to match the set of lower and upper case letters, \. to match periods, 
# and \s to match whitespace between words.
# Use str.contains() to create a Boolean vector representing values that match the pattern.
# Invert the mask by placing a ~ before it.
# Subset the countries series using the .loc[] accessor and mask_inverse. 
# Then hit 'Submit Answer' to see the invalid country names!


gapminder = pd.read_csv('gapminder.csv')
# Create the series of countries: countries
countries = gapminder['country']

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
pattern = r'^[A-Za-z\.\s]*$'

# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse
mask_inverse = ~mask

# Subset countries using mask_inverse: invalid_countries
invalid_countries = countries.loc[mask_inverse]

# Print invalid_countries
print(invalid_countries)
