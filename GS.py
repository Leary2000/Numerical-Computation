import numpy as np
import scipy.linalg as la

def gauss_seidel_relaxation(A, b, omega, tolerance=1e-10, max_iterations=100):
    m, n = A.shape
    x = b / np.diagonal(A)
    for iteration in range(max_iterations):
        x_new = np.copy(x)
        for i in range(m):
            sum = b[i]
            for j in range(n):
                if i != j:
                    sum -= A[i][j] * x_new[j]
            x_new[i] = (1 - omega) * x[i] + (omega / A[i][i]) * sum

        # Check for convergence
        if np.allclose(x, x_new, atol=tolerance, rtol=0.):
            return x_new, iteration + 1
        x = x_new
    return x, max_iterations

def gauss_seidel(A, b, tolerance=1e-10, max_iterations=100):
    x = np.zeros_like(b, dtype=np.double)
    
    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(len(b)):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new
        x = x_new
    return x


def jacobi(A, b, tolerance=1e-10, max_iterations=100):
    x = np.zeros_like(b, dtype=np.double)
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    for i in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new
        x = x_new
    return x

A = np.array([[4, -1, 0, -1, 0, 0, 0, 0, 0],
              [-1, 4, -1, 0, -1, 0, 0, 0, 0],
              [0, -1, 4, 0, 0, -1, 0, 0, 0],
              [-1, 0, 0, 4, -1, 0, -1, 0, 0],
              [0, -1, 0, -1, 4, -1, 0, -1, 0],
              [0, 0, -1, 0, -1, 4, 0, 0, -1],
              [0, 0, 0, -1, 0, 0, 4, -1, 0],
              [0, 0, 0, 0, -1, 0, -1, 4, -1],
              [0, 0, 0, 0, 0, -1, 0, -1, 4]])

b = np.array([4, -1, -5, -2, 2, 2, -1, 1, 6])

# Solving using Jacobi and Gauss-Seidel methods
x_jacobi = jacobi(A, b)
x_gauss_seidel = gauss_seidel(A, b)

print("Solution using Jacobi:", x_jacobi)
print("Solution using Gauss-Seidel:", x_gauss_seidel)

def jacobi_iteration_matrix(A):
    D = np.diag(A)
    R = A - np.diagflat(D)
    D_inv = np.diag(1 / D)
    return -np.dot(D_inv, R)

def gauss_seidel_iteration_matrix(A):
    n = len(A)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    M = la.solve(-D - L, U)
    return M

# Calculate iteration matrices
M_jacobi = jacobi_iteration_matrix(A)
M_gauss_seidel = gauss_seidel_iteration_matrix(A)

# Calculate norms
norm_M_jacobi = np.linalg.norm(M_jacobi, ord=np.inf)
norm_M_gauss_seidel = np.linalg.norm(M_gauss_seidel, ord=np.inf)

# Calculate eigenvalues
eigvals_jacobi = np.linalg.eigvals(M_jacobi)
eigvals_gauss_seidel = np.linalg.eigvals(M_gauss_seidel)

print("Jacobi Iteration Matrix M:", M_jacobi)
print("Gauss-Seidel Iteration Matrix M:", M_gauss_seidel)
print("Norm of M (Jacobi):", norm_M_jacobi)
print("Norm of M (Gauss-Seidel):", norm_M_gauss_seidel)
print("Eigenvalues of M (Jacobi):", eigvals_jacobi)
print("Eigenvalues of M (Gauss-Seidel):", eigvals_gauss_seidel)

# Check convergence
spectral_radius_jacobi = max(abs(eigvals_jacobi))
spectral_radius_gauss_seidel = max(abs(eigvals_gauss_seidel))

print("Spectral radius (Jacobi):", spectral_radius_jacobi)
print("Spectral radius (Gauss-Seidel):", spectral_radius_gauss_seidel)
print("Convergence (Jacobi):", spectral_radius_jacobi < 1)
print("Convergence (Gauss-Seidel):", spectral_radius_gauss_seidel < 1)






# Tridiagonal
def create_tridiagonal(n):
    # Creating a tridiagonal matrix of size n with 2 on the diagonal and -1 on off-diagonals
    main_diag = 2 * np.ones(n)
    off_diag = -1 * np.ones(n-1)
    return np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

n = 100
A = create_tridiagonal(n)

# Calculating the norm of A
norm_A = la.norm(A, 2)  # Spectral norm (2-norm)

# Calculating the eigenvalues of A
eigenvalues = la.eigvals(A)

# Output results
print("Norm of A:", norm_A)
print("Eigenvalues of A:", eigenvalues)

# Comment on the results
print("The spectral norm of A gives a sense of the largest stretch factor of the matrix.")
print("The eigenvalues are indicative of the stability and conditioning of the matrix.")