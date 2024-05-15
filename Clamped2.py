import numpy as np
import matplotlib.pyplot as plt

# Given points
x_points = np.array([1, 2, 3, 4])
y_points = np.array([4, 8, 9, 9])

# Boundary conditions
S_prime_1 = 1
S_prime_4 = -2

# Calculate h values
h = np.diff(x_points)

# Calculate d values
d = np.diff(y_points) / h

# Calculate u values
u = 6 * np.diff(d)

print("h values:", h)
print("d values:", d)
print("u values:", u)

# Form the linear system
A = np.array([
    [2 * (h[0] + h[1]), h[1]],
    [h[1], 2 * (h[1] + h[2])]
])

B = np.array([
    u[0] - 3 * ((y_points[1] - y_points[0]) / h[0] - S_prime_1),
    u[1] - 3 * (S_prime_4 - (y_points[3] - y_points[2]) / h[2])
])

# Solve the linear system
m_internal = np.linalg.solve(A, B)
print("m_internal:", m_internal)

# Compute m1 and m4 using boundary conditions
m1 = (3 * ((y_points[1] - y_points[0]) / h[0] - S_prime_1) - m_internal[0]) / 2
m4 = (3 * (S_prime_4 - (y_points[3] - y_points[2]) / h[2]) - m_internal[1]) / 2

# Full m array including boundary conditions
m = np.concatenate(([-4.3809], [-0.4766], [0.40635], [-2]))

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
x_range = np.linspace(1, 4, 1000)
y_range = [cubic_spline(x, x_points, a, b, c, d) for x in x_range]

# Plot the spline and the data points
plt.plot(x_range, y_range, label='Clamped Cubic Spline')
plt.scatter(x_points, y_points, color='red', label='Data Points')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.legend()
plt.title('Clamped Cubic Spline')
plt.grid(True)
plt.savefig('/home/conor/crypto/Clamped2.png')
plt.close() 

# Print the coefficients for each spline segment
for i in range(len(a)):
    print(f"S_{i+1}(x) = {a[i]:.2f}(x - {x_points[i]})^3 + {b[i]:.2f}(x - {x_points[i]})^2 + {c[i]:.2f}(x - {x_points[i]}) + {d[i]:.2f} for {x_points[i]} <= x < {x_points[i+1]}")

# Verify boundary conditions
print(f"S'_1(1) = {c[0]:.2f} (should be 1)")
print(f"S'_3(4) = {3*a[2]*(4-3)**2 + 2*b[2]*(4-3) + c[2]:.2f} (should be -2)")
