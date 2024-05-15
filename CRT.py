from sympy.ntheory.modular import crt
import numpy as np
remainders = [0, 5, 37]
moduli = [4, 9, 43]

solution, _ = crt(moduli, remainders)
s = solution % 26316

print("CRT solution modulo the product of moduli:", solution)
print("Solution x[1] modulo 26316:", s)

mod_product = 4 * 9 * 43  # 1548

# To find the k that gives 19688
for k in range(20):  # Test a reasonable range of k values
    test_x = 1112 + k * mod_product
    if test_x % 26316 == 19688 % 26316:
        print(f"Found matching k: {k}, x: {test_x}")
        break

# Define the augmented matrix
augmented_matrix = np.array([
    [0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 28717],
    [0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 2612],
    [0, 3, 2, 0, 0, 1, 0, 0, 0, 0, 2371],
    [0, 0, 3, 0, 1, 0, 1, 0, 0, 0, 11957],
    [0, 1, 0, 3, 1, 0, 0, 0, 0, 0, 8368],
    [3, 0, 0, 0, 0, 0, 1, 1, 0, 0, 9978],
    [2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 9203],
    [0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 11275],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 9257],
    [0, 5, 0, 1, 0, 0, 0, 0, 0, 0, 30226],
    [0, 1, 2, 0, 1, 0, 0, 0, 1, 0, 4885],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 22566],
    [0, 1, 1, 2, 0, 0, 0, 1, 0, 0, 2468],
    [0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 11513],
    [2, 2, 2, 0, 0, 1, 0, 0, 0, 0, 23705],
    [5, 0, 2, 0, 0, 0, 0, 1, 0, 0, 11874],
    [2, 1, 1, 0, 1, 0, 0, 0, 1, 0, 32110],
    [0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 7255],
    [2, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1245],
    [1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 3783],
    [1, 3, 1, 1, 0, 0, 0, 0, 0, 0, 12000],
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 7607],
    [0, 3, 2, 0, 0, 0, 0, 0, 0, 1, 8637]
])

# Split the matrix into A and B
A = augmented_matrix[:, :-1]
B = augmented_matrix[:, -1]

# Function to apply Gaussian elimination in modular arithmetic
def solve_modular(A, B, mod):
    n, m = A.shape  # n is the number of rows, m is the number of columns
    A = A.copy().astype(int)
    B = B.copy().astype(int)
    x = np.zeros(m, dtype=int)  # Solution vector should match the number of columns in A

    for i in range(min(n, m)):  # Only go up to the smaller of n or m
        # Find pivot for column i in row i or below
        max_row = i + np.argmax(np.abs(A[i:n, i]) % mod)
        if A[max_row, i] == 0 or np.gcd(A[max_row, i], mod) != 1:
            continue  # Skip if no valid pivot (non-invertible under mod)

        # Swap if necessary
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            B[i], B[max_row] = B[max_row], B[i]

        # Normalize row i
        inv = pow(int(A[i, i]), -1, mod)
        A[i] = (A[i] * inv) % mod
        B[i] = (B[i] * inv) % mod

        # Eliminate all other entries in this column
        for j in range(n):
            if j != i:
                factor = A[j, i]
                A[j] = (A[j] - factor * A[i]) % mod
                B[j] = (B[j] - factor * B[i]) % mod

    # Solve from the last row up
    for i in range(m-1, -1, -1):
        sum_ax = 0
        for j in range(i+1, m):
            sum_ax += A[i, j] * x[j]
            sum_ax %= mod
        x[i] = (B[i] - sum_ax) % mod

    return x

# Solve the system modulo 17
x_mod_17 = solve_modular(A, B, 17)

# Calculate the Euclidean norm of the solution
norm = np.linalg.norm(x_mod_17)

print("Solution vector modulo 17:", x_mod_17)
print("Euclidean norm of the solution:", norm)
