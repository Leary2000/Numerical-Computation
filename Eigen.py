import numpy as np
from scipy.sparse.linalg import eigs
import scipy.sparse as sparse
# Define a square matrix
A = np.array([[4, 1, 2],
              [2, 3, 1],
              [1, 2, 4]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)



# Example: Creating a large sparse matrix
size = 1000  # Size of the matrix
A_sparse = sparse.diags(A, shape=(size, size), format='csr')

# Find the 3 largest eigenvalues and their corresponding eigenvectors
eigenvalues, eigenvectors = eigs(A_sparse, k=3, which='LM')

print("Eigenvalues:", eigenvalues)


