import pandas as pd 


########## multilevel concat
for medal in medal_types:

    file_name = "%s_top5.csv" % medal

    # Read file_name into a DataFrame: medal_df
    medal_df = pd.read_csv(file_name, index_col='Country')

    # Append medal_df to medals
    medals.append(medal_df)

# Concatenate medals: medals
medals = pd.concat(medals, keys=['bronze', 'silver', 'gold'], axis='rows')

# Print medals in entirety
print(medals)


############## Slicing multileveled indexed dataframe using pd.IndexSlice
# Sort the entries of medals: medals_sorted
medals_sorted = medals.sort_index(level=0)

# Print the number of Bronze medals won by Germany
print(medals_sorted.loc[('bronze', 'Germany')])

# Print data about silver medals
print(medals_sorted.loc['silver'])

# Create alias for pd.IndexSlice: idx
idx = pd.IndexSlice

# Print all the data on medals won by the United Kingdom
print(medals_sorted.loc[idx[:, 'United Kingdom'], :])
# pek1 = medals_sorted.loc[idx[:,'United Kingdom'], :]
# pek2 = medals_sorted.loc[(slice(None), slice('United Kingdom','United Kingdom')), :]

# print((pek1 == pek2).all())


# the slice(start, end) must have two arguments or None as a wildcard


# Concatenate dataframes: february
february = pd.concat(
    dataframes, keys=['Hardware', 'Software', 'Service'], axis=1)

# Print february.info()
print(february.info())

# Assign pd.IndexSlice: idx
idx = pd.IndexSlice

# Create the slice: slice_2_8
slice_2_8 = february.loc['Feb 2, 2015':'Feb 8, 2015', idx[:, 'Company']]

# Print slice_2_8
print(slice_2_8)


# Your task is to aggregate the sum of all sales over the 'Company' column into
# a single DataFrame. You'll do this by constructing a dictionary of these DataFrames
# and then concatenating them.


# Make the list of tuples: month_list
month_list = [('january', jan), ('february', feb),  ('march', mar)]

# Create an empty dictionary: month_dict
month_dict = {}

for month_name, month_data in month_list:

    # Group month_data: month_dict[month_name]
    month_dict[month_name] = month_data.groupby('Company').sum()


# Concatenate data in month_dict: sales
# sales = pd.concat(month_dict, axis=1)
sales = pd.concat(month_dict)
# Print sales
print(sales)

# Print all sales by Mediacore
idx = pd.IndexSlice
print(sales.loc[idx[:, 'Mediacore'], :])
# print(sales.loc['Mediacore',])
