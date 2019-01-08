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
