import numpy as np

def qr_decomposition_verify(A):
    # Performing QR decomposition
    Q, R = np.linalg.qr(A, mode='complete')
    return Q, R, np.dot(Q, R)

A = np.array([[1, 1], [2, 1], [3, 1]])  # Example matrix that might be decomposed

# Decompose and verify
Q, R, A_reconstructed = qr_decomposition_verify(A)

print("Matrix Q:\n", Q)
print("Matrix R:\n", R)
print("Reconstructed A:\n", A_reconstructed)

# Verify that Q is orthogonal and R is upper triangular
# ans = top row of Q
print("Is Q orthogonal?:", np.allclose(np.dot(Q.T, Q), np.identity(Q.shape[1])))
print("Is R upper triangular?:", np.allclose(R, np.triu(R)))



# A is a square Matrix

import numpy as np

def solve_with_qr(A, B):
    # Step 1: QR factorization of A
    Q, R = np.linalg.qr(A)
    
    # Step 2: Compute Q^T * B
    QTB = np.dot(Q.T, B)
    
    # Step 3: Solve RX = QTB for X using back substitution
    X = np.linalg.solve(R, QTB)
    
    return X

# Example usage
# Define matrix A and vector B
A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])
B = np.array([1, 0, 1])

# Solve the system AX = B
X = solve_with_qr(A, B)

print("Solution X:", X)





# A not Square (Least Squares Problem)
print()
def solve_least_squares(A, B):
    # Step 1: QR factorization of A
    Q, R = np.linalg.qr(A)
    
    # Step 2: Compute Q^T * B
    QTB = np.dot(Q.T, B)
    
    # Step 3: Solve RX = QTB for X using the upper triangular part of R
    # and the corresponding part of QTB
    X = np.linalg.solve(R[:A.shape[1], :], QTB[:A.shape[1]])
    
    return X

# Example usage
A = np.array([[1, 1], [2, 1], [3, 2]])
B = np.array([2, 1, 1])

# Solve the least squares problem AX = B
X = solve_least_squares(A, B)

# Check the solution
AX = np.dot(A, X)
residual_norm = np.linalg.norm(AX - B)

print("Solution X:", X)
print("Norm of AX-B:", residual_norm)
