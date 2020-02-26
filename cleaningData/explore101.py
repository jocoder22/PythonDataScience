import numpy as np
import pandas as pd 


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# download World Bank population data
worldpop = pd.read_excel("http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel", 
                       index_col=None, skiprows=3)




