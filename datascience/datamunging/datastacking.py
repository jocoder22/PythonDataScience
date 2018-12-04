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

# working on 3-dimension, using dstack
data9 = data1.copy()
data9[:5, :3] = 200
data9[:5, 3:] = 300
data9[5:10, :3] = 400
data9[5:10, 3:] = 500
data9[10:15, :3] = 600
data9[10:15, 3:] = 700
data9[15:, :3] = 800
data9[15:, 3:] = 900


data10 = np.dstack((data1, data9))
print(data1.shape, data9.shape, data10.shape)
print(data1.ndim, data9.ndim, data10.ndim)
