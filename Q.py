import numpy as np

def main():
    c = np.array([2, 1, 1, 3])
    n = len(c)
    A = np.zeros((n, n))
    for i in range(n):
        A[:, i] = np.roll(c, i)
    
    frobenius_norm_A = np.linalg.norm(A, 'fro')
    print("i) Frobenius norm of A:", round(frobenius_norm_A, 2))
    print(A)
    
    I_n = np.identity(n)
    A_half = A / 2
    matrix_part_ii = np.linalg.matrix_power(I_n + A_half, 2)
    frobenius_norm_part_ii = np.linalg.norm(matrix_part_ii, 'fro')
    print("ii) Frobenius norm of (I_n + A/2)^2:", round(frobenius_norm_part_ii, 2))
    
    k = 1000000
    A_k = A / k
    matrix_part_iii = np.linalg.matrix_power(I_n + A_k, k)
    frobenius_norm_part_iii = np.linalg.norm(matrix_part_iii, 'fro')
    print("iii) Frobenius norm of (I_n + A/k)^k:", round(frobenius_norm_part_iii, 2))

if __name__ == "__main__":
    main()
