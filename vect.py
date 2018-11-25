#!/usr/bin/env/python 
 
import sys 
from datetime import datetime 
import numpy as np 
 
""" 
This program demonstrates vector addition the Python way. 
Run the following from the command line: 
 
  python vectorsum.py n 
 
Here, n is an integer that specifies the size of the vectors. 
 
The first vector to be added contains the squares of 0 up to n. 
The second vector contains the cubes of 0 up to n. 
The program prints the last 2 elements of the sum and the elapsed  time: 
""" 
 
def numpysum(n): 
   a1 = np.arange(n) ** 2 
   b1 = np.arange(n) ** 3 
   c1 = a1 + b1 
 
   return c1 
 
def pythonsum(n): 
   a = list(range(n)) 
   b = list(range(n)) 
   c = [] 
 
   for i in range(n): 
       a[i] = i ** 2 
       b[i] = i ** 3 
       c.append(a[i] + b[i]) 
 
   return c 
 


size = int(sys.argv[1]) 
print("Processing for n = ", size)
start = datetime.now() 
c2 = pythonsum(size) 
delta = datetime.now() - start 
print("The last 2 elements of the sum", c2[-2:]) 
print("PythonSum elapsed time in microseconds", delta.microseconds) 
 
start = datetime.now() 
c3 = numpysum(size) 
delta = datetime.now() - start 
print("The last 2 elements of the sum", c3[-2:]) 
print("NumPySum elapsed time in microseconds", delta.microseconds) 




size = int(sys.argv[2]) 
print()
print()
print("Processing for n = ", size)

start = datetime.now() 
c = pythonsum(size) 
delta = datetime.now() - start 
print("The last 2 elements of the sum", c[-2:]) 
print("PythonSum elapsed time in microseconds", delta.microseconds) 
 
start = datetime.now() 
c = numpysum(size) 
delta = datetime.now() - start 
print("The last 2 elements of the sum", c[-2:]) 
print("NumPySum elapsed time in microseconds", delta.microseconds) 




size = int(sys.argv[3]) 
print()
print()
print("Processing for n = ", size) 
start = datetime.now() 
c = pythonsum(size) 
delta = datetime.now() - start 
print("The last 2 elements of the sum", c[-2:]) 
print("PythonSum elapsed time in microseconds", delta.microseconds) 
 
start = datetime.now() 
c = numpysum(size) 
delta = datetime.now() - start 
print("The last 2 elements of the sum", c[-2:]) 
print("NumPySum elapsed time in microseconds", delta.microseconds) 

