import numpy as np
import matplotlib.pyplot as plt

# Given points
x_points = np.array([0, 1, 2, 3])
y_points = np.array([0, 0.5, 2.0, 1.5])

# Calculate h values
h = np.diff(x_points)

# Calculate d values
d = np.diff(y_points) / h

# Calculate u values
u = 6 * np.diff(d)

print("h values:", h)
print("d values:", d)
print("u values:", u)

# Number of intervals
n = len(h)

# Form the matrix A and vector B for the linear system to find m2 and m3
A = np.zeros((n-1, n-1))
B = np.zeros(n-1)

A[0, 0] = 2 * (h[0] + h[1])
A[0, 1] = h[1]
B[0] = u[0]

A[1, 0] = h[1]
A[1, 1] = 2 * (h[1] + h[2])
B[1] = u[1]

print("Matrix A:")
print(A)
print("Vector B:")
print(B)

# Solve the linear system for m2 and m3
m_internal = np.linalg.solve(A, B)
print("m_internal:", m_internal)

# Set boundary conditions for natural cubic spline
m1 = 0
m4 = 0

# Full m array including boundary conditions
m = np.concatenate(([m1], m_internal, [m4]))

print("m values:", m)

# Calculate the coefficients of the spline segments
def spline_coefficients(x, y, m, h):
    a = (m[1:] - m[:-1]) / (6 * h)
    b = m[:-1] / 2
    c = (y[1:] - y[:-1]) / h - (2 * h * m[:-1] + h * m[1:]) / 6
    d = y[:-1]
    return a, b, c, d

a, b, c, d = spline_coefficients(x_points, y_points, m, h)

# Define the piecewise cubic spline functions
def cubic_spline(x, x_points, a, b, c, d):
    n = len(x_points) - 1
    for i in range(n):
        if x_points[i] <= x < x_points[i+1]:
            dx = x - x_points[i]
            return a[i]*dx**3 + b[i]*dx**2 + c[i]*dx + d[i]
    return a[-1]*(x - x_points[-2])**3 + b[-1]*(x - x_points[-2])**2 + c[-1]*(x - x_points[-2]) + d[-1]

# Evaluate the spline at a range of x values
x_range = np.linspace(0, 3, 100)
y_range = [cubic_spline(x, x_points, a, b, c, d) for x in x_range]

# Plot the spline and the data points
plt.plot(x_range, y_range, label='Natural Cubic Spline')
plt.scatter(x_points, y_points, color='red', label='Data Points')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.legend()
plt.title('Natural Cubic Spline')
plt.grid(True)
plt.savefig('/home/conor/crypto/NaturalSpline.png')
plt.close() 

# Print the coefficients for each spline segment
for i in range(len(a)):
    print(f"S_{i+1}(x) = {a[i]:.2f}(x - {x_points[i]})^3 + {b[i]:.2f}(x - {x_points[i]})^2 + {c[i]:.2f}(x - {x_points[i]}) + {d[i]:.2f} for {x_points[i]} <= x < {x_points[i+1]}")
