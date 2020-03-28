#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sp = '\n\n'
# path = 'C:\\Users\\Jose\\Desktop\\PythonDataScience\\Projects\\datacamps\\crypto\\'
path = "D:\PythonDataScience\Projects\datacamps\crypto"
os.chdir(path)
data = pd.read_csv('crypto.csv')

# print(data.head(), data.shape, data.columns, data.info(), sep=sp)

# Selecting the 'id' and the 'market_cap' columns
market_cap = data[['name','market_cap', 'circulating_supply', 'total_supply',
       'max_supply']]
print(market_cap.head(), end=sp)

# Counting the number of values
print(market_cap.count(), end=sp)

# Filtering out rows without a market capitalization
cap = market_cap.query('market_cap > 0')

# Counting the number of values again
print(cap.count(), end=sp)


#Declaring these now for later use in the plots
TOP_CAP_TITLE = 'Top 10 market capitalization'
LOW_CAP_TITLE = 'Bottom 10 market capitalization'
CAP_LABEL = '% of total cap'

# Selecting the first 10 rows and setting the index
cap10 = cap.loc[:9,:].set_index('name')

# Calculating market_cap_perc
cap10 = cap10.assign(market_cap_perc = lambda x: (x.market_cap / cap.market_cap.sum()) * 100)
cap10.head()

# Plotting the barplot with the title defined above 
ax = cap10['market_cap_perc'].plot.bar(title=TOP_CAP_TITLE)

# Annotating the y axis with the label defined above
ax.set_ylabel(CAP_LABEL)
ax.set_xlabel("")
plt.show()


# Plotting the barplot with the title defined above 
ax = cap10['market_cap_perc'].plot.barh(title=TOP_CAP_TITLE)

# Annotating the y axis with the label defined above
ax.set_xlabel(CAP_LABEL)
ax.set_ylabel("")
plt.show()



# Calculating market_cap_perc
cap_10R = cap.assign(market_cap_perc = lambda x: (x.market_cap / cap.market_cap.sum()) * 100)
cap_10R.sort_values('market_cap_perc', ascending=False, inplace=True)
cap_10R = cap_10R.iloc[-10:,:].set_index('name')
cap_10R.head()


# Plotting the barplot with the title defined above 
ax = cap_10R['market_cap_perc'].plot.barh(title=LOW_CAP_TITLE)
# 'circulating_supply', 'total_supply', 'max_supply' are varible for future analysis

# Annotating the y axis with the label defined above
ax.set_xlabel(CAP_LABEL)
ax.set_ylabel("")
ax.set_xscale('log')
plt.show()


plt.style.use('fivethirtyeight')
# Colors for the bar plot
COLORS = ['orange', 'green', 'purple', 'red', 'cyan', 'blue', 'silver', 'yellow', 'indigo', 'pink']

# Plotting market_cap as before but adding the colors and scaling the y-axis  
ax = cap10['market_cap_perc'].plot.bar(title=TOP_CAP_TITLE, color=COLORS, logy=True)

# Annotating the y axis with 'USD'.
ax.set_ylabel('USD')

# Final touch! Removing the xlabel as it is not very informative
ax.set_xlabel("")
plt.show()


# Selecting the id, percent_change_24h and percent_change_7d columns
volatility = data[['name', 'percent_change_24h', 'percent_change_7d']]

# Setting the index to 'id' and dropping all NaN rows
volatility = volatility.set_index('name').dropna()

# Sorting the DataFrame by percent_change_24h in ascending order
volatility = volatility.sort_values('percent_change_24h', ascending=True)

# # Checking the first few rows
print(volatility.head())


data2 = list(data['name'].values)
data2 = sorted(data2)
print(data2)


#Defining a function with 2 parameters, the series to plot and the title
def top10_subplot(volatility_series, title):
    # Making the subplot and the figure for two side by side plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    # Plotting with pandas the barchart for the top 10 losers
    ax = (volatility_series[:10].plot.bar(color="darkred", title='Top Losers', ax=axes[0]))
    
    # Setting the figure's main title to the text passed as parameter
    fig.suptitle(title)
    
    # Setting the ylabel to '% change'
    ax.set_ylabel('% change')
    ax.set_xlabel("")
    
    # Same as above, but for the top 10 winners
    ax = (volatility_series[-10:].plot.bar(color="darkblue", title='Top  Winners', ax=axes[1]))
    ax.set_xlabel("")

DTITLE = "24 hours top losers and winners"

# Calling the function above with the 24 hours period series and title DTITLE  
top10_subplot(volatility.percent_change_24h, DTITLE)
plt.show()


# Sorting in ascending order
volatility7d = volatility.sort_values('percent_change_7d', ascending=True)

WTITLE = "Weekly top losers and winners"

# Calling the top10_subplot function
top10_subplot(volatility7d.percent_change_7d, WTITLE)
plt.show()

large_caps = cap.query('market_cap > 10000000000')
print(large_caps.head())


# Large cap: +10 billion
# Mid cap: 2 billion - 10 billion
# Small cap: 300 million - 2 billion
# Micro cap: 50 million - 300 million
# Nano cap: Below 50 million

def capcount(query_string):
    return cap.query(query_string).name.count()

# Labels for the plot
LABELS = ["biggish", "mid", "small", "micro", "nano"]

# Using capcount count the biggish cryptos
biggish = capcount('market_cap >= 10000000000')

# Same as above for mid ...
mid = capcount('market_cap>=2000000000 & market_cap<=10000000000')

# Same as above for small ...
small = capcount('market_cap>=300000000 & market_cap<=2000000000')

# Same as above for micro ...
micro = capcount('market_cap>=50000000 & market_cap<=300000000')

# ... and for nano
nano =  capcount('market_cap<50000000')

# Making a list with the 3 counts
values = [biggish, mid, small, micro, nano]


# Plotting them with matplotlib 
plt.bar(LABELS, values)
for idx, key in enumerate(values):
       ypoint = key + 10
       plt.text(idx, ypoint , key, color='black', va='bottom', ha='center', fontweight='bold')
plt.show()
print(values, sep=sp)