import numpy as np
from scipy.linalg import hilbert, solve

# Create the Hilbert matrix H(3) and vector b
H = hilbert(3)
b = np.array([1, 0, 0])

# (i) Solve the system using exact arithmetic
exact_solution = solve(H, b)

# Helper function for two-digit chopped representation
def chop_matrix(M, digits=2):
    factor = 10**digits
    return (M * factor).astype(int) / factor

# (ii) Chop the Hilbert matrix to two digits
H_chopped = chop_matrix(H)

# Function to perform Gaussian elimination
def gaussian_elimination(A, b, partial_pivoting=False):
    n = len(b)
    M = A.astype(float)
    b = b.astype(float)

    for i in range(n):
        if partial_pivoting:
            # Partial pivoting
            max_row = np.argmax(np.abs(M[i:, i])) + i
            M[[i, max_row]], b[i], b[max_row] = M[[max_row, i]], b[max_row], b[i]

        # Elimination
        for j in range(i+1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(M[i, i+1:], x[i+1:])) / M[i, i]
    return x

# (iii) Solve without partial pivoting
solution_no_pp = gaussian_elimination(H_chopped, b, partial_pivoting=False)

# (iv) Solve with partial pivoting
solution_with_pp = gaussian_elimination(H_chopped, b, partial_pivoting=True)

# (v) Solve the chopped system using exact arithmetic
exact_chopped_solution = solve(H_chopped, b)

print("Exact solution:", exact_solution)
print("Chopped matrix:\n", H_chopped)
print("Solution without P.P.:", solution_no_pp)
print("Solution with P.P.:", solution_with_pp)
print("Exact solution of chopped system:", exact_chopped_solution)
