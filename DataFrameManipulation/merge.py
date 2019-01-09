import pandas as pd 

######### merge
# Merge revenue and sales: revenue_and_sales
revenue_and_sales = pd.merge(
    revenue, sales, how='right',  on=['city', 'state'])

# Print revenue_and_sales
print(revenue_and_sales)

# Merge sales and managers: sales_and_managers
sales_and_managers = pd.merge(sales, managers, how='left', left_on=[
                              'city', 'state'], right_on=['branch', 'state'])

# Print sales_and_managers
print(sales_and_managers)


# Perform the first merge: merge_default
merge_default = pd.merge(sales_and_managers, revenue_and_sales)

# Print merge_default
print(merge_default)

# Perform the second merge: merge_outer
merge_outer = pd.merge(sales_and_managers, revenue_and_sales, how='outer')

# Print merge_outer
print(merge_outer)

# Perform the third merge: merge_outer_on
merge_outer_on = pd.merge(sales_and_managers, revenue_and_sales, on=[
                          'city', 'state'], how='outer')

# Print merge_outer_on
print(merge_outer_on)


################# merge_ordered
# Merge auto and oil: merged
merged = tx_weather = pd.merge_asof(auto, oil, left_on='yr', right_on='Date')

# Print the tail of merged
print(merged.tail())

# Resample merged: yearly
yearly = merged.resample('A', on='Date')[['mpg', 'Price']].mean()

# Print yearly
print(yearly)

# print yearly.corr()
print(yearly.corr())
