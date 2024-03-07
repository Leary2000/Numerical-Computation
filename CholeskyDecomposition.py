# Cholesky decomposition is a special case of LU decomposition where the matrix is symmetric and positive definite

import numpy as np

def cholesky_decomposition3x3(A):
    """Performs Cholesky decomposition on a 3x3 matrix."""
    L = np.zeros_like(A)
    
    # Compute the Cholesky decomposition
    L[0, 0] = np.sqrt(A[0, 0])
    L[1, 0] = A[1, 0] / L[0, 0]
    L[2, 0] = A[2, 0] / L[0, 0]
    
    L[1, 1] = np.sqrt(A[1, 1] - L[1, 0]**2)
    L[2, 1] = (A[2, 1] - L[2, 0] * L[1, 0]) / L[1, 1]
    
    L[2, 2] = np.sqrt(A[2, 2] - L[2, 0]**2 - L[2, 1]**2)
    
    return L

# Example usage
A = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]], dtype=float)

L = cholesky_decomposition3x3(A)
print("L:\n", L)
print("L * L.T:\n", np.dot(L, L.T))




import numpy as np

def cholesky_decomposition(A):
    """Performs Cholesky decomposition on a square, positive definite matrix."""
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i+1):
            sum = 0
            if j == i:  # Diagonal elements
                for k in range(j):
                    sum += L[j, k] ** 2
                L[j, j] = np.sqrt(A[j, j] - sum)
            else:
                for k in range(j):
                    sum += L[i, k] * L[j, k]
                L[i, j] = (A[i, j] - sum) / L[j, j]
    return L

# Example usage with a larger matrix
np.random.seed(0)  # For reproducibility
A = np.random.rand(50, 50)
A = np.dot(A, A.T)  # Make A symmetric and positive definite

L = cholesky_decomposition(A)
print("L:\n", L)
print("Verification (L * L.T):\n", np.dot(L, L.T))
print("Original Matrix A:\n", A)

# For very large matrices, like 1000x1000, the same function can be used:
# Be cautious as this might be slow and consume a lot of memory for very large matrices.
# A_large = np.random.rand(1000, 1000)
# A_large = np.dot(A_large, A_large.T)  # Make A_large symmetric and positive definite
# L_large = cholesky_decomposition(A_large)

