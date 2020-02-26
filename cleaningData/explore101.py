import numpy as np
import pandas as pd 


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# download World Bank population data
worldpop = pd.read_excel("http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel", 
                       index_col=None, skiprows=3)



print2(worldpop.shape, worldpop[worldpop.isnull().any(axis=1)], worldpop.columns, worldpop.info())

worldpop2 = worldpop.drop(columns=["2019"])

print2(worldpop2.info(), worldpop2.isnull().sum())

print("China" in worldpop['Country Name'].values)