#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

sp = '\n\n'
path = 'C:\\Users\\Jose\\Desktop\\PythonDataScience\\Projects\\datacamps\\crypto\\'
os.chdir(path)
data = pd.read_csv('crypto.csv')

print(data.head(), data.shape, data.columns, data.info(), sep=sp)

# Selecting the 'id' and the 'market_cap' columns
market_cap = data[['name','market_cap']]
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

# Annotating the y axis with the label defined above
ax.set_xlabel(CAP_LABEL)
ax.set_ylabel("")
plt.show()
