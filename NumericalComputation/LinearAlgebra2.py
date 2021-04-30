# import sympy 
from sympy import *
init_printing(use_unicode=True)

# form Matrix
Ma = Matrix([[1, 3, 9, 4], [2, 6, 5,-2], [4, 1, 3, -7], [5, 0, 4, 6]])
Mb = Matrix([[2, 3, 2, -1], [-2, -1, 5,-2], [2, 0, 3, -1], [5, -2, 2, 4]])

# Matrix addition and substration : + , -

Ma + Mb
Ma - Mb

# Matrix Multiplication: *

Ma * Mb

# self Multiplicaton
Ma*Ma
Ma**2

# rows and columns: Zero Indexed
Ma.row(1)
Ma.col(1)

# find the shape
Ma.shape

# find inverse: **-1
Ma**-1
Mb**-1

# Create common matrices, Identity => eye(), zero => zero(), onces => onces(), diagonal => diag()
eye(3)
zeros(3,4)
ones(2,5)
diag(3,4,7,0)

# Determinant
det(Ma)

# Reduced row echelon
Ma.rref()
