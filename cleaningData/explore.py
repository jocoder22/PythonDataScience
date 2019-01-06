#!/usr/bin/env python
# Import pandas
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