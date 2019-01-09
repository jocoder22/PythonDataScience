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


############## inner joins
# Create the list of DataFrames: medal_list
medal_list = [bronze, silver, gold]

# Concatenate medal_list horizontally using an inner join: medals
medals = pd.concat(medal_list, join='inner', axis=1)
medals = pd.concat(medal_list, join='inner', axis=1).sum(axis=1)
medals = pd.concat(medal_list, keys=[
                   'bronze', 'silver', 'gold'], join='inner', axis=1).sum(axis=1)
medals = pd.concat(medal_list, keys=[
                   'bronze', 'silver', 'gold'], join='inner', axis=1).sum(axis=0)

# Print medals
print(medals)

# Resample and tidy china: china_annual
china_annual = china.resample('A').pct_change(10).dropna()

# Resample and tidy us: us_annual
us_annual = us.resample('A').pct_change(10).dropna()

# Concatenate china_annual and us_annual: gdp
gdp = pd.concat([china_annual, us_annual], axis=1, join='inner')

# Resample gdp and print
print(gdp.resample('10A').last())


# Add 'state' column to revenue: revenue['state']
revenue['state'] = ['TX', 'CO', 'IL', 'CA']

# Add 'state' column to managers: managers['state']
managers['state'] = ['TX', 'CO', 'CA', 'MO']

# Merge revenue & managers on 'branch_id', 'city', & 'state': combined
combined = pd.merge(revenue, managers, on=['branch_id', 'city', 'state'])

# Print combined
print(combined)
