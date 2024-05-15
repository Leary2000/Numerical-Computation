from CholeskyDecomposition import cholesky_decomposition
import numpy as np


# Forward substitution
def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b)

    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y

# Backward substitution
def backward_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y)
    print("U: \n",U)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

# The matrices A and b
A = np.array([[1, 1, 1],
              [1, 2, 2],
              [1, 2, 3]], dtype=float)
b = np.array([1, 1.5, 3], dtype=float)

# Perform Cholesky decomposition
L = cholesky_decomposition(A)

# Solve Ly=b
y = forward_substitution(L, b)
# Solve L^Tx=y
x = backward_substitution(L.T, y)

# Verify the solution
verification = np.dot(A, x)

# Outputs
print("Cholesky factor L:\n", L)
print("Solution x:\n", x)
# Check if verfication and b matrix are the same
print("Verification (Ax):\n", verification)
print("B Matrix :\n", b)
