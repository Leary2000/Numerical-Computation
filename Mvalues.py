import numpy as np

def gauss_seidel(A, b, tolerance=1e-10, max_iterations=100):
    x = np.zeros_like(b, dtype=np.double)
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(len(b)):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new
        x = x_new
    return x

# Given values
h = np.array([1, 1, 1])
d = np.array([0.5, 1.5, -0.5])
S_prime_0 = 0.2
S_prime_3 = -1.0

# Boundary conditions for m0 and mn based on first derivatives
m0 = 6 * ((d[0] - 0) / h[0] - S_prime_0) / h[0]
mn = 6 * (S_prime_3 - (d[-1] - d[-2]) / h[-1]) / h[-1]

# Construct the tridiagonal matrix A and vector b for m1, m2, and m3
n = len(h) + 1  # Number of internal points plus boundary conditions

A = np.zeros((n, n))
b = np.zeros(n)

# Fill the tridiagonal matrix A and vector b
for i in range(1, n - 1):
    A[i, i - 1] = h[i - 1]
    A[i, i] = 2 * (h[i - 1] + h[i - 1])
    A[i, i + 1] = h[i - 1]
    b[i] = 6 * ((d[i] - d[i - 1]) / h[i - 1] - (d[i - 1] - d[i - 2]) / h[i - 2])

# Set boundary conditions
A[0, 0] = 1
A[-1, -1] = 1
b[0] = m0
b[-1] = mn

# Print A and b for verification
print("Matrix A:\n", A)
print("Vector b:\n", b)

# Solve using Gauss-Seidel method
m_values = gauss_seidel(A, b)
print("m values:", m_values)
