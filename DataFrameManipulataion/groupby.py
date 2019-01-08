import pandas as pd 

############# groupby
# Group titanic by 'pclass'
by_class = titanic.groupby('pclass')

# Aggregate 'survived' column of by_class by count
# Number of those that survived by passengers class
count_by_class = by_class.survived.count()

# Print count_by_class
print(count_by_class)

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked', 'pclass'])

# Aggregate 'survived' column of by_mult by count
# Number of passengers that survived by embarked  and class
count_mult = by_mult.survived.count()

# Print count_mult
print(count_mult)


# Read life_fname into a DataFrame: life
life = pd.read_csv(life_fname, index_col='Country')

# Read regions_fname into a DataFrame: regions
regions = pd.read_csv(regions_fname, index_col='Country')

# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions['region'])

# Print the mean over the '2010' column of life_by_region
print(life_by_region['2010'].mean())

# Group titanic by 'pclass': by_class
by_class = titanic.groupby('pclass')

# Select 'age' and 'fare'
by_class_sub = by_class[['age', 'fare']]

# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(['max', 'median'])

# Print the maximum age in each class
print(aggregated.loc[:, ('age', 'max')])

# Print the median fare in each class
print(aggregated.loc[:, ('fare', 'median')])


# Read the CSV file into a DataFrame and sort the index: gapminder
gapminder = pd.read_csv('gapminder.csv', index_col=[
                        'Year', 'region', 'Country']).sort_index()

# Group gapminder by 'Year' and 'region': by_year_region
by_year_region = gapminder.groupby(level=['Year', "region"])

# Define the function to compute spread: spread


def spread(series):
    return series.max() - series.min()


# Create the dictionary: aggregator
aggregator = {'population': 'sum', 'child_mortality': 'mean', 'gdp': spread}

# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)

# Print the last 6 entries of aggregated
aggregated.tail(6)
print(aggregated.tail(6))
