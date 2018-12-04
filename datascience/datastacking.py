import numpy as np

data1 = np.arange(12*10).reshape(20, 6)
data2 = np.arange(60).reshape(10, 6)
data3 = np.arange(6).reshape(1, 6)

# vertical stacking using vstack, data grows long, vertically
# ## here we are adding more values
np.vstack((data1, data2))
np.vstack((data1, data3, data2))


# Horinzontial stacking using hstack, data grows wide, horizontally
# ## here we are adding more variables
# ## can also use column_stack()
data4 = np.arange(20).reshape(20, 1)
data5 = np.arange(20)
np.hstack((data1, data4))
np.column_stack((data1, data5))

