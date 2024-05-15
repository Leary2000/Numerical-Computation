import numpy as np
import scipy.linalg  # For LU decomposition
import timeit

# The matrix A
A = np.array([[1., 1., 0.],
              [1., 2., 1.],
              [0., 1., 3.]])

# LU Decomposition-based inverse function
def inverse_LU(matrix):
    P, L, U = scipy.linalg.lu(matrix)
    inv = scipy.linalg.inv(U) @ scipy.linalg.inv(L) @ P.T
    return inv

# Built-in inverse function
def inverse_builtin(matrix):
    return np.linalg.inv(matrix)

# Timing function
def time_function(func, *args):
    start_time = timeit.default_timer()
    for _ in range(1000):  # Repeat the operation 1000 times
        func(*args)
    end_time = timeit.default_timer()
    return end_time - start_time

# Timing the computations
time_LU = time_function(inverse_LU, A)
time_builtin = time_function(inverse_builtin, A)

# Displaying the results
print(f"A is {A}")
print(f"Inverse of A is {inverse_builtin(A)}")
print(f"Timings over 1000 iterations:")
print(f"Standard np based inverse time taken: {time_builtin:.5f} seconds")
print(f"LU based time taken: {time_LU:.5f} seconds")
print(f"Standard based is: {time_LU / time_builtin:.4f} times faster")

