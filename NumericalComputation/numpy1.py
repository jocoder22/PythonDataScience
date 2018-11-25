import numpy as np

x = np.array([[2, 2, 5, 6], [6, 8, 10, 4]])
print(x)
print("We've a", type(x))
print("The array has a shape of", x.shape)
print("Our array has total size of", x.size)
print("with dimension of", x.ndim)
print("The data type is", x.dtype)
print("Memory size is", x.nbytes, "bytes")


# compare memory sizes
xint32 = np.array([[2, 2, 5, 6], [6, 8, 10, 4]], dtype=np.int32)
print(xint32, "Memory size of", xint32.nbytes, "bytes")
xint64 = np.array([[2, 2, 5, 6], [6, 8, 10, 4]], dtype=np.int64)
print(xint64, "Memory size of", xint64.nbytes, "bytes")
xfloat = np.array([[2, 2, 5, 6], [6, 8, 10, 4]], dtype=np.float)
print(xfloat, "Memory size of", xfloat.nbytes, "bytes")
xcomplex = np.array([[2, 2, 5, 6], [6, 8, 10, 4]], dtype=np.complex)
print(xcomplex, "Memory size of", xcomplex.nbytes, "bytes")
xuint32 = np.array([[2, 2, 5, 6], [6, 8, 10, 4]], dtype=np.uint32)
print(xuint32, "Memory size of", xuint32.nbytes, "bytes")


# changing dtypes
y = np.array([[4, 9, 12], [2, 7, 11]])
yfloat = y.astype(np.float)
yint64 = np.array(y, dtype=np.int64)

print(yfloat.dtype, "with memory size of", yfloat.nbytes, "bytes")
print(y.dtype, "with memory size of", y.nbytes, "bytes")
print(yint64.dtype, "with memory size of", yint64.nbytes, "bytes")


# large dataset
largeData = np.random.rand(1000000, 100)
print(type(largeData))
print(largeData.dtype)
print("largeData has", largeData.nbytes, "bytes of memory size")
floatLargeData = largeData.astype(np.float32)
print("floatLargeData has", floatLargeData.nbytes, "bytes of memory size")
