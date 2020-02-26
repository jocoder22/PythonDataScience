import numpy as np
import pandas as pd 


# download World Bank population data
worldpop = pd.read_excel("http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=excel", 
                       index_col=None, skiprows=3)




