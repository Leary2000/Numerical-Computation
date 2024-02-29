from sympy import Matrix, symbols, linsolve
import numpy as np
from scipy.linalg import lu_factor, lu_solve

# Define the symbols
x1, x2, x3 = symbols('x1 x2 x3')

# Define the system using SymPy's Rational type to ensure exact arithmetic
Hilbert = Matrix([[1, 1/2, 1/3, 1/4 ,1/5],
                [1/2, 1/3, 1/4, 1/5, 1/6],
                [1/3, 1/4, 1/5, 1/6, 1/7],
                [1/4, 1/5, 1/6, 1/7, 1/8],
                [1/5, 1/6, 1/7, 1/8, 1/9]])

b_sym = Matrix([1, 0, 0])

# Solve the system using exact arithmetic
x_sym = linsolve((Hilbert, b_sym), x1, x2, x3)

print("Solved using exact arithmetic: \n", x_sym)




# Function to chop numbers to two digits
def chop_to_two_digits(matrix):
    return np.array([[float('%.2f' % element) for element in row] for row in matrix])

# Define the Hilbert matrix A using two-digit chopped representation
A = np.array([[1, 1/2, 1/3],
              [1/2, 1/3, 1/4],
              [1/3, 1/4, 1/5]])
A_chopped = chop_to_two_digits(A)

# Define the vector b
b = np.array([1, 0, 0])

# Solve the system using numpy's linalg.solve, which does not use partial pivoting
x_no_pp = np.linalg.solve(A_chopped, b)

print("Solved using numpy and no PP: \n", x_no_pp )




# Perform LU decomposition with partial pivoting
lu, piv = lu_factor(A_chopped)

# Solve the system using the LU decomposition
x_with_pp = lu_solve((lu, piv), b)
print("Solved using scipy and PP: \n", x_with_pp )


from sympy import Matrix, linsolve, symbols

x1, x2, x3 = symbols('x1 x2 x3')

# Convert A and b to SymPy matrices for exact arithmetic
Hilbert = Matrix(A.tolist())
b_sym = Matrix(b.tolist())

# Solve the system using linsolve for exact solutions
x_exact = linsolve((Hilbert, b_sym), x1, x2, x3)

print("Solved using linsolve: \n", x_exact)