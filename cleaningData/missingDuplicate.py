import numpy as np 
import pandas as pd


billboard = pd.read_csv('billboard.csv')

# Create the new DataFrame: tracks
tracks = billboard[['year', 'artist', 'track', 'time']]

# Print info of tracks
print(tracks.info())

# Drop the duplicates: tracks_no_duplicates
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks
print(tracks_no_duplicates.info())



# Read in the dataset
airquality = pd.read_csv('airquality.csv')

# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality.Ozone.mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality.Ozone.fillna(oz_mean)

# Print the info of airquality
print(airquality.info())







# Note: You can use pd.notnull(df) as an alternative to df.notnull().
# When using these within an assert statement, nothing will be returned 
# if the assert statement is true: This is how you can confirm that the 
# data you are checking are valid.


# Assert that there are no missing values
assert pd.notnull(airquality).all().all()

# Assert that all values are >= 0
assert (airquality >= 0).all().all()


def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()
    numeric = pd.to_numeric(no_na)
    ge0 = numeric >= 0
    return ge0




gapminder = pd.read_csv('gapminder.csv')

# Drop the missing values
# Drop the rows in the data where any observation in life_expectancy is missing. 
# As confirmed that country and year don't have missing values, we can 
# use the .dropna() method on the entire gapminder DataFrame, because any missing 
# values would have to be in the life_expectancy column. The .dropna() method has 
# the default keyword arguments axis=0 and how='any', which specify that rows 
# with any missing values should be dropped.

gapminder = gapminder.dropna(how='any')

# Print the shape of gapminder
print(gapminder.shape)