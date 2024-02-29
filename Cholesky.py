import numpy as np
from scipy.linalg import cholesky

# Set the size of the matrix
n = 1000  # create a 100x100 matrix

# Create a random matrix
A = np.random.rand(n, n)

# Make the matrix symmetric
A = A @ A.T

# Add a multiple of the identity to make A positive definite
A = A + n*np.identity(n)

# Perform Cholesky decomposition using numpy

# numpy.linalg.cholesky returns the lower triangular matrix L
L = np.linalg.cholesky(A)

# Display the result
print("Lower triangular matrix L:\n", L)

# To verify the decomposition, you can check that A = L @ L^T
print("Check that A = L @ L^T:\n", L @ L.T)


# Using scipy
L = cholesky(A, lower = True)
print("Lower triangular matrix L:\n", L)
print("Check that A = L @ L^T:\n", L @ L.T)
