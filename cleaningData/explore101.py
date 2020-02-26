import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# download World Bank population data
worldpop = pd.read_excel("http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel", 
                       index_col=None, skiprows=3)


# explore the data
print2(worldpop.shape, worldpop[worldpop.isnull().any(axis=1)], worldpop.columns, worldpop.info())

# No data for 2019, so drop the column
worldpop2 = worldpop.drop(columns=["2019"])

print2(worldpop2.info(), worldpop2.isnull().sum(), worldpop2[worldpop2.isnull().any(axis=1)])

print2("China" in worldpop['Country Name'].values)

chinadata = worldpop2.loc[worldpop2['Country Name'] == "China", "1960":]

# print the list
print2(chinadata)

plt.plot(chinadata.columns, chinadata.values[0])
plt.xticks(rotation=45)
plt.grid()
plt.show()