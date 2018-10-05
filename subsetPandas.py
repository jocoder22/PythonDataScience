import pandas as pd
from pandas import Series, DataFrame
import numpy as np

ser100 = Series(np.arange(6),
                index=['Tv', 'Radio', 'Cellphone',
                       'Laptop', 'Monitor', 'Heater'])

# Subsetting;
ser100[:2]
ser100[['Radio', 'Monitor']]
ser100['Radio': 'Monitor']
ser100[ser100 > 3]
ser100 > 3


# index selection;
ser200 = Series(['Zero', 'One', 'Two', 'Three', 'Four'],
                index=[3, 2, 1, 0, 4])

ser200[2:4] 
"""
1      Two
0    Three
dtype: object
"""

ser200.iloc[2:4]  
"""
1      Two
0    Three
dtype: object
"""

ser200.loc[2:4]  
"""
2      One
1      Two
0    Three
4     Four
dtype: object
"""
