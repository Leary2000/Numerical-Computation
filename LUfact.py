import numpy as np
from scipy.linalg import lu_factor, lu_solve

def lu_decomposition_solve():
    A = np.array([[-3, 2, -1],
                  [6, -6, 7],
                  [3, -4, 4]])
    B = np.array([1, 0, 1])

    # Factorizing the matrix A
    lu, piv = lu_factor(A)

    # Solving the system Ax = B using the LU decomposition
    x = lu_solve((lu, piv), B)

    return x, lu, piv

def verify_lu_decomposition(A, lu, piv):
    # Extract L and U from lu
    L = np.tril(lu, k=-1) + np.eye(A.shape[0])
    U = np.triu(lu)
    
    # Verify that A = LU
    A_reconstructed = np.dot(L, U)
    
    return np.allclose(A, A_reconstructed), L, U

x, lu, piv = lu_decomposition_solve()

print("Solution x:", x)
print("LU matrix (packed):\n", lu)
print("Pivot indices:", piv)

is_correct, L, U = verify_lu_decomposition(np.array([[-3, 2, -1], [6, -6, 7], [3, -4, 4]]), lu, piv)

print("Is A = LU?:", is_correct)
print("L matrix:\n", L)
print("U matrix:\n", U)
