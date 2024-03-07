import numpy as np
import scipy.linalg
import time

# Define the square matrix A
A = np.array([[1., 1., 0., 5., 2.],
              [1., 2., 1., 2., 3.],
              [0., 1., 3., 1., 0.],
              [1., 4., 5., 2., 1.],
              [3., 5., 7., 1., 4.]])

# Function to compute inverse using LU decomposition
def inverse_using_lu(matrix):
    P, L, U = scipy.linalg.lu(matrix)
    inv_U = np.linalg.inv(U)
    inv_L = np.linalg.inv(L)
    inv_P = np.linalg.inv(P)
    return inv_U @ inv_L @ inv_P

# Function to compute inverse using NumPy's built-in function
def inverse_using_numpy(matrix):
    return np.linalg.inv(matrix)

# Time the LU decomposition method
start_time = time.time()
for _ in range(1000):
    inv_A_lu = inverse_using_lu(A)
lu_time_taken = time.time() - start_time

# Time the NumPy built-in inverse function
start_time = time.time()
for _ in range(1000):
    inv_A_np = inverse_using_numpy(A)
np_time_taken = time.time() - start_time

# Display results
print("A is", A)
print("inverse of A is", inv_A_np)
print("Timings over 1000 iterations")
print("Standard np based inverse time taken:", np_time_taken, "seconds")
print("LU based time taken:", lu_time_taken, "seconds")

# Calculate and display which method is faster and by how much
# Numpy method is generally faster
if np_time_taken < lu_time_taken:
    print("Standard based is:", lu_time_taken / np_time_taken, "times faster")
else:
    print("LU based is:", np_time_taken / lu_time_taken, "times faster")
