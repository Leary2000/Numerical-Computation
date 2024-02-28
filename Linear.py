import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import minimize



def solve_linear_system(A, b):
    x = np.linalg.solve(A, b)
    return x

def invert_matrix(A):
    A_inv = np.linalg.inv(A)
    return A_inv

def calculate_determinant(A):
    det = np.linalg.det(A)
    return det

def euclidean_norm(v):
    norm = np.linalg.norm(v)
    return norm

def frobenius_norm(A):
    norm = np.linalg.norm(A, 'fro')
    return norm

def scalar_matrix_multiplication(scalar, A):
    return scalar * A

def matrix_power(A, n):
    return np.linalg.matrix_power(A, n)

def transpose_matrix(A):
    return A.T

def func(x):
    return x**2 - 6*x + 9


    # Define the function F(x)
def F(x):
    return np.sin(x**2) + x**2 - 2*x - 0.09


if __name__ == "__main__":

    A = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])
    x = solve_linear_system(A, b)
    print("Solution of the system:", x)

    A_inv = invert_matrix(A)
    det = calculate_determinant(A)
    print("Inverse of A:", A_inv)
    print("Determinant of A:", det)


    A = np.array([[1, 2], [3, 4]])
    scalar = 2
    A_scaled = scalar_matrix_multiplication(scalar, A)
    A_squared = matrix_power(A, 2)
    A_transposed = transpose_matrix(A)

    print("Scalar * Matrix:\n", A_scaled)
    print("Matrix squared:\n", A_squared)
    print("Transpose of Matrix:\n", A_transposed)

    norm_A = frobenius_norm(A)
    print("Frobenius norm of A:", norm_A)

    v = np.array([3, 4])
    norm_v = euclidean_norm(v)
    print("Euclidean norm of v:", norm_v)

    A = np.array([[3, 1], [1, 2]])
    b = np.array([9, 8])
    x = solve_linear_system(A, b)
    print("Solution of the system:", x) 

    #Need to pass in function method : x**2 - 6*x + 9
   # sol = root_scalar(func, bracket=[0, 10], method='brentq')

    #print(f"Root of the function: {sol.root}")


    # Initial guess
    x0 = [0]

    # Perform the minimization
    res = minimize(func, x0)

    print(f"Maximum of the function at: {res.x[0]}")





    # Use root_scalar with the bracket method, providing a bracket around x = 2
    # We choose a small interval around 2 as we're told the root is around x = 2
    sol = root_scalar(F, bracket=[0.6, 1.6])

    # Print the root, rounded to two decimal places
    print(f"The root of F(x) around x = 2 is: {sol.root:.2f}")
