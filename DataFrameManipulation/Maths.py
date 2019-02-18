import pandas as pd

######## maths
# Extract selected columns from weather as new DataFrame: temps_f
temps_f = weather[['Min TemperatureF',
                   'Mean TemperatureF', 'Max TemperatureF']]

# temps_f = weather.loc[:, ['Min TemperatureF',
#                    'Mean TemperatureF', 'Max TemperatureF']]

# Convert temps_f to celsius: temps_c
temps_c = (temps_f - 32) * 5/9

# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F', 'C')

# Print first 5 rows of temps_c
print(temps_c.head())


########## percentage change
gdp = pd.read_csv('GDP.csv', parse_dates=True, index_col='DATE')

# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp.loc['2008':, ]

# Print the last 8 rows of post2008
print(post2008.tail(8))

# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample('A').last()

# Print yearly
print(yearly)

# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change() * 100

# Print yearly again
print(yearly)


# Read 'sp500.csv' into a DataFrame: sp500
sp500 = pd.read_csv('sp500.csv', parse_dates=True, index_col='Date')

# Read 'exchange.csv' into a DataFrame: exchange
exchange = pd.read_csv('exchange.csv', parse_dates=True, index_col='Date')

# Subset 'Open' & 'Close' columns from sp500: dollars
# dollars = sp500.loc[:, ['Open', 'Close']]
dollars = sp500[['Open', 'Close']]

# Print the head of dollars
print(dollars.head())

# Convert dollars to pounds: pounds
# pounds = dollars.multiply(exchange['GBP/USD'], axis='rows')
pounds = dollars.mul(exchange['GBP/USD'], axis='rows')

# Print the head of pounds
print(pounds.head())


data = pd.read_csv(
    "https://cdncontribute.geeksforgeeks.org/wp-content/uploads/nba.csv")
