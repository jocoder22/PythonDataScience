import numpy as np
import pandas as pd
from pandas import Series, DataFrame


sp = {'sep':'\n\n', 'end':'\n\n'}

multiindex = pd.MultiIndex([['AA', 'BB'], ['Male', 'Female'],
                            ['Youth', 'Old']],
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
              #    index = ["bgroup", "sex", "Agecat"],
                 columns=['Age', 'Weight', 'Height'])

print(data.index, data, data.loc['AA'],
data.loc[('AA', 'Female')],
data.loc[('AA', 'Female', 'Old')], **sp)


# Using slicer;
print(
data.loc[('AA', slice(None), 'Old')],
data.loc[('AA', slice(None), 'Old'), :],
data.loc[('AA', slice(None), 'Old'), 'Age'],
data.loc[('AA', slice(None, 'Male'), 'Old')],
data.loc[('AA', slice(None, 'Male'), 'Old'), 'Age'],
data.loc[(slice(None), slice(None), 'Old'), 'Age'],
data.loc[(slice(None, 'BB'), slice(None, 'Male'), 'Old'), 'Age'], **sp)



print(data.loc['AA']['Age'].sum)
