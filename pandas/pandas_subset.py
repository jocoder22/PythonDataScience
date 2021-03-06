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


dframe = DataFrame(np.arange(15).reshape(5, 3),
                   columns=["Price", "Discount", "Sales"],
                   index=["AAA", "BBB", "CCC", "DDD", "EEE"])

dframe.Price  # column selection
dframe.Sales  # column selection
dframe['Sales']  # column selection
dframe['Price']  # column selection


# More subsetting;
dframe[['Price', 'Sales']]
dframe.iloc[1:4, 0:2]
dframe.iloc[1:3, 1:2]
dframe.loc['CCC':'EEE', 'Price':'Sales']
dframe.iloc[:, 1:3]

# Mixing types;
dframe.iloc[:, 1:3].loc[['AAA', 'DDD']]  # double pass selection
dframe2 = dframe.iloc[:, 1:3].loc[['AAA', 'DDD']].copy()
dframe2
"""
     Discount  Sales
AAA         1      2
DDD        10     11
"""


# Changing values of dataframe;
dframe2['Discount'] = Series({'AAA': 9, 'DDD': 8})
dframe2
"""
     Discount  Sales
AAA         9      2
DDD         8     11
"""

dframe2.iloc[0,0] =12
dframe2
"""
     Discount  Sales
AAA        12      2
DDD         8     11
"""

dframe2.iloc[:,1] = 6
dframe2
"""
     Discount  Sales
AAA        12      6
DDD         8      6
"""