import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection='3d')

t = np.arange(30)

x = -1 - t
y = 2 - t
z = 7 + 3*t

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label', labelpad = 20.0)
ax.set_ylabel('Y Label', labelpad = 20.0)
ax.set_zlabel('Z Label')


plt.show()
