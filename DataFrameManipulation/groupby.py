from scipy.stats import zscore
import pandas as pd 

# download titanic dataset
url = 'https://assets.datacamp.com/production/repositories/502/datasets/e280ed94bf4539afb57d8b1cbcc14bcf660d3c63/titanic.csv'
titan = pd.read_csv(url, sep=',')
titan.to_csv('titanic.csv')


# download the olympic medalist dataset
url2 = 'https://assets.datacamp.com/production/repositories/502/datasets/bf22326ecc9171f68796ad805a7c1135288120b6/all_medalists.csv'
meda = pd.read_csv(url2, sep=',')
meda.to_csv('medals.csv')


url3 = 'https://assets.datacamp.com/production/repositories/502/datasets/09378cc53faec573bcb802dce03b01318108a880/gapminder_tidy.csv'
gapmd = pd.read_csv(url3, sep=',')
gapmd.to_csv('gapminder.csv')

############# groupby
# Group titanic by 'pclass'
titanic = pd.read_csv('titanic.csv')
by_class = titanic.groupby('pclass')

# Aggregate 'survived' column of by_class by count
# Number of those that survived by passengers class
print("######################################")
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


# # Read life_fname into a DataFrame: life
# life = pd.read_csv(life_fname, index_col='Country')

# # Read regions_fname into a DataFrame: regions
# regions = pd.read_csv(regions_fname, index_col='Country')

# # Group life by regions['region']: life_by_region
# life_by_region = life.groupby(regions['region'])

# # Print the mean over the '2010' column of life_by_region
# print(life_by_region['2010'].mean())

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


'''
# Read file: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Create a groupby object: by_day
by_day = sales.groupby(sales.index.strftime('%a'))

# Create sum: units_sum
units_sum = by_day['Units'].sum()

# Print units_sum
print(units_sum)


# In this example, you're going to normalize the Gapminder data in 2010 for life expectancy
# and fertility by the z-score per region. Using boolean indexing, you will filter out
# countries that have high fertility rates and low life expectancy for their region.
############### transform
# Import zscore

# Group gapminder_2010: standardized
standardized = gapminder_2010.groupby(
    'region')['life', 'fertility'].transform(zscore)

# Construct a Boolean Series to identify outliers: outliers
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

# Filter gapminder_2010 by the outliers: gm_outliers
gm_outliers = gapminder_2010.loc[outliers]

# Print gm_outliers
print(gm_outliers)



# Using the olympic dataset
# Select the 'NOC' column of medals: country_names
medals = pd.read_csv('medals.csv')
country_names = medals['NOC']

# Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()

# Print top 15 countries ranked by medals
print(medal_counts.head(15))


# Construct the pivot table: counted
counted = medals.pivot_table(
    index='NOC', columns='Medal', values='Athlete', aggfunc='count')

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis='columns')

# Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending=False)

# Print the top 15 rows of counted
print(counted.head(15))


# In this exercise, you will use .pivot_table() first to aggregate the total medals by type.
# Then, you can use .sum() along the columns of the pivot table to produce a new column.
# When the modified pivot table is sorted by the total medals column,
# you can display the results from the last exercise with a bit more detail.
# Select columns: ev_gen
ev_gen = medals[['Event_gender', 'Gender']]

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)


# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender', 'Gender'])

# Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()

# Print medal_count_by_gender
print(medal_count_by_gender)


# Create the Boolean Series: sus
sus = (medals.Event_gender == 'W') & (medals.Gender == 'Men')

# Create a DataFrame with the suspicious row: suspect
suspect = medals[sus]

# Print suspect
print(suspect)


# Group medals by 'NOC': country_grouped
country_grouped = medals.groupby('NOC')

# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped.Sport.nunique()

# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending=False)

# Print the top 15 rows of Nsports
print(Nsports.head(15))

# Extract all rows for which the 'Edition' is between 1952 & 1988: during_cold_war
during_cold_war = (medals.Edition >= 1952) & (medals.Edition <= 1988)

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA', 'URS'])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)

# Print Nsports
print(Nsports)
'''