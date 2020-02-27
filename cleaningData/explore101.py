import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import requests


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# download World Bank population data
worldpop = pd.read_excel("http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel", 
                       index_col=None, skiprows=3)


# explore the data
print2(worldpop.shape, worldpop[worldpop.isnull().any(axis=1)], worldpop.columns, worldpop.info())


# columns with at least one missing value
# explore the data
print2(worldpop.isnull().any().sum(), worldpop.shape[1])

# No data for 2019, so drop the column
worldpop2 = worldpop.drop(columns=["2019"])

# show the column with any missing value
print2(worldpop2[worldpop2.isnull().any(axis=1)])


# Number of columns with all missing value
print2(worldpop.isnull().all().sum(), worldpop.shape[1])


# show the column with all missing value
print2(worldpop[worldpop.isnull().all(axis=1)])



# Number of rows with all missing value
print2(worldpop.isnull().all(axis=1).sum(), worldpop.shape[1])
# show the column with all missing value
print2(worldpop[worldpop.isnull().all(axis=1)])


# explore the data
print2(worldpop.shape, worldpop[worldpop.isnull().all(axis=1)], worldpop.columns, worldpop.info())



print2(worldpop2.info(), worldpop2.isnull().sum(), worldpop2[worldpop2.isnull().any(axis=1)])

print2("China" in worldpop['Country Name'].values)

chinadata = worldpop2.loc[worldpop2['Country Name'] == "China", "1960":]

# print the list
print2(chinadata)

plt.plot(chinadata.columns, chinadata.values[0])
plt.xticks(rotation=45)
plt.grid()
plt.show()


######################################################################
# working with json files
# download json file using API
url = 'http://api.worldbank.org/v2/countries/br;cn;us;de/indicators/SP.POP.TOTL/?format=json&per_page=1000'
r = requests.get(url)
print2(r.json()[1][218])

# create pandas dataframe, bad way
df = pd.DataFrame(r.json()[1])
print2(df.head())


# create pandas dataframe, good way
mydict = {"Indicator": [], "Country": [], "Year": [], "Population": []}

for i in r.json()[1]:
    mydict["Indicator"].append(i["indicator"]["value"].split(",")[0]+"_Total")
    mydict["Country"].append(i["country"]["value"])
    mydict["Year"].append(i["date"])
    mydict["Population"].append(i["value"])

df2 = pd.DataFrame.from_dict(mydict)
print2(df2.head())
