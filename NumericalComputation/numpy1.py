import numpy as np 

x = np.array([[2, 2, 5, 6], [6, 8, 10, 4]])
print(x)
print("We've a", type(x))
print("The array has a shape of", x.shape)
print("Our array has total size of", x.size)
print("with dimension of", x.ndim)
print("The data type is", x.dtype)
print("Memory size is", x.nbytes, "bytes")
