import pandas as pd 

# Read the CSV file into a DataFrame: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company.Units.sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
# by_com_filt = by_com_sum.loc[by_com_sum > 35]
by_com_filt = by_company.filter(lambda g: g['Units'].sum() > 35)
print(by_com_filt)


# In this exercise your job is to investigate survival rates of passengers on the Titanic by
# 'age' and 'pclass'. In particular, the goal is to find out what fraction of children under
# 10 survived in each 'pclass'. You'll do this by first creating a boolean array where True is
#  passengers under 10 years old and False is passengers over 10. You'll use .map() to change
#  these values to strings.

# Finally, you'll group by the under 10 series and the 'pclass' column and aggregate the
# 'survived' column. The 'survived' column has the value 1 if the passenger survived and 0
#  otherwise. The mean of the 'survived' column is the fraction of passengers who lived.

# Create the Boolean Series: under10
under10 = (titanic['age'] < 10).map({True: 'under 10', False: 'over 10'})

# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
print(survived_mean_1)

# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10, 'pclass'])['survived'].mean()
print(survived_mean_2)


# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama': 'blue', 'Romney': 'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election['winner'].map(red_vs_blue)

# Print the output of election.head()
print(election.head())
