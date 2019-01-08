import pandas as pd 
from scipy.stats import zscore

###################### pivotting
# Pivot the users DataFrame: visitors_pivot

visitors_pivot = users.pivot(
    index='weekday', columns='city', values='visitors')

# Print the pivoted DataFrame
print(visitors_pivot)

# Pivot the users DataFrame with the 'signups' indexed by 'weekday'
# in the rows and 'city' in the columns.
# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot = users.pivot(index='weekday', columns='city', values='signups')

# Print signups_pivot
print(signups_pivot)


# Pivot the users DataFrame with both 'signups' and 'visitors' pivoted
# - that is, all the variables. This will happen automatically if you do not
# specify an argument for the values parameter of .pivot().
# Pivot users pivoted by both signups and visitors: pivot
pivot = users.pivot(index='weekday', columns='city')

# Print the pivoted DataFrame
print(pivot)


######################### pivot table
# Create the DataFrame with the appropriate pivot table: by_city_day
by_city_day = users.pivot_table(index='weekday', columns='city')

# Print by_city_day
print(by_city_day)


# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index='weekday', aggfunc='count')

# Print count_by_weekday
print(count_by_weekday1)

# Replace 'aggfunc='count'' with 'aggfunc=len': count_by_weekday2
count_by_weekday2 = users.pivot_table(index='weekday', aggfunc=len)

# Verify that the same result is obtained
print('==========================================')
print(count_by_weekday1.equals(count_by_weekday2))


# Create the DataFrame with the appropriate pivot table: signups_and_visitors
signups_and_visitors = users.pivot_table(index='weekday', aggfunc=sum)

# Print signups_and_visitors
print(signups_and_visitors)

# Add in the margins: signups_and_visitors_total
signups_and_visitors_total = users.pivot_table(
    index='weekday', aggfunc=sum, margins=True)

# Print signups_and_visitors_total
print(signups_and_visitors_total)
