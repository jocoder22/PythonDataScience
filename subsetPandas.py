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

