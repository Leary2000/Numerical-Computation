import numpy as np

# Use for non square matrices
# Define a matrix A 100 x 100
A = np.random.rand(100, 100)

# Perform QR decomposition
Q, R = np.linalg.qr(A)

# Display the results
print("Orthogonal matrix Q:\n", Q)
print("Upper triangular matrix R:\n", R)

# To verify the decomposition, you can check that A = Q @ R
# Q @ R is the matrix multiplication of Q and R
print("Check that A = Q @ R:\n", np.allclose(A, Q @ R))
