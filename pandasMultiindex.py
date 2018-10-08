import numpy as np
import pandas as pd
from pandas import Series, DataFrame

multiindex = pd.MultiIndex([['AA', 'BB'], ['Male', 'Female'], ['Youth', 'Old']],
                           [[0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1]])
                
Series(np.arange(8), index=multiindex)

ser10 = Series(np.arange(8),
               index=[['AA', 'AA', 'AA', 'AA', 'CC', 'CC', 'CC', 'CC'],
                      ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
                      [1, 2, 1, 2, 1, 2, 1, 2]])

ser10.loc['AA']
ser10.loc['AA', 'F']
ser10.loc['AA', 'F', 2]
ser10.loc['AA', :, 1]
ser10.loc[:, :, 2]
ser10.loc['AA', :, :]


# multiindex dataframe;
data = DataFrame(np.random.randn(8, 3), index=multiindex,
                 columns=['Age', 'Weight', 'Height'])



                