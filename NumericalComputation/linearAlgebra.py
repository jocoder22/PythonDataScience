import numpy as np
from numpy.linalg import multi_dot 

# Dot product
x1 = np.arange(20).reshape(5, 4)
x2 = np.arange(12).reshape(4, 3)
x3 = np.arange(18).reshape(3, 6)
x4 = np.arange(12).reshape(6, 2)

np.dot(x1, x2)
np.matmul(x1, x2)
x1@x2

multi_dot([x1, x2])
multi_dot([x1, x2, x3, x4])

# inner and outer products
y = np.arange(12).reshape(4, 3)
z = np.arange(3)

np.inner(y, z)
np.outer(y, z)


y = np.arange(12)
np.ndim(y)  # this is one dimension now
np.outer(y, z)

# higher dimension product
a = np.arange(12).reshape(2,3,2) 
b = np.arange(48).reshape(3,2,8) 
c = np.tensordot(a,b, axes =([1,0],[0,1])) 
print(a) 
print(b)
print(c)