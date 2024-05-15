import numpy as np
import matplotlib.pyplot as plt

# Given points
x_points = np.array([0, 1, 2, 3])
y_points = np.array([0, 0.5, 2.0, 1.5])

# Boundary conditions for clamped spline
S_prime_0 = 0.2
S_prime_3 = -1.0

# Calculate h values
h = np.diff(x_points)

# Calculate d values
d = np.diff(y_points) / h

# Calculate u values
u = 6 * np.diff(d)

# Print known values of h, d, and u
print("h values:", h)
print("d values:", d)
print("u values:", u)

# Form the linear system for the clamped spline
A_clamped = np.array([
    [3.5, 1.0],
    [1.0, 3.5]
])

B_clamped = np.array([
    5.1,
    -10.5
])

# Solve the linear system for m2 and m3 (clamped)
m_clamped_internal = np.linalg.solve(A_clamped, B_clamped)
print("m_clamped_internal:", m_clamped_internal)

# Compute m1 and m4 using boundary conditions for clamped spline
m1_clamped = 3 * ((y_points[1] - y_points[0]) / h[0] - S_prime_0) - m_clamped_internal[0] / 2
m4_clamped = 3 * (S_prime_3 - (y_points[3] - y_points[2]) / h[2]) - m_clamped_internal[1] / 2

# Full m array including boundary conditions (clamped)
m_clamped = np.concatenate(([m1_clamped], m_clamped_internal, [m4_clamped]))
print("m_clamped values:", m_clamped)

# Set boundary conditions for natural cubic spline
m1_natural = 0
m4_natural = 0

# Form the matrix A and vector B for the natural spline
A_natural = np.array([
    [2 * (h[0] + h[1]), h[1]],
    [h[1], 2 * (h[1] + h[2])]
])

B_natural = np.array([
    u[0],
    u[1]
])

# Solve the linear system for m2 and m3 (natural)
m_natural_internal = np.linalg.solve(A_natural, B_natural)
print("m_natural_internal:", m_natural_internal)

# Full m array including boundary conditions (natural)
m_natural = np.concatenate(([m1_natural], m_natural_internal, [m4_natural]))
print("m_natural values:", m_natural)

# Function to calculate the spline coefficients
def spline_coefficients(x, y, m, h):
    a = (m[1:] - m[:-1]) / (6 * h)
    b = m[:-1] / 2
    c = (y[1:] - y[:-1]) / h - (2 * h * m[:-1] + h * m[1:]) / 6
    d = y[:-1]
    return a, b, c, d

# Calculate coefficients for clamped and natural splines
a_clamped, b_clamped, c_clamped, d_clamped = spline_coefficients(x_points, y_points, m_clamped, h)
a_natural, b_natural, c_natural, d_natural = spline_coefficients(x_points, y_points, m_natural, h)

# Define the piecewise cubic spline function
def cubic_spline(x, x_points, a, b, c, d):
    n = len(x_points) - 1
    for i in range(n):
        if x_points[i] <= x < x_points[i+1]:
            dx = x - x_points[i]
            return a[i]*dx**3 + b[i]*dx**2 + c[i]*dx + d[i]
    return a[-1]*(x - x_points[-2])**3 + b[-1]*(x - x_points[-2])**2 + c[-1]*(x - x_points[-2]) + d[-1]

# Evaluate the spline at a denser range of x values
x_range = np.linspace(0, 3, 1000)
y_clamped_range = [cubic_spline(x, x_points, a_clamped, b_clamped, c_clamped, d_clamped) for x in x_range]
y_natural_range = [cubic_spline(x, x_points, a_natural, b_natural, c_natural, d_natural) for x in x_range]

# Plot the splines and the data points
plt.plot(x_range, y_clamped_range, label='Clamped Cubic Spline', color='blue')
plt.plot(x_range, y_natural_range, label='Natural Cubic Spline', color='red')
plt.scatter(x_points, y_points, color='green', label='Data Points')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.legend()
plt.title('Clamped and Natural Cubic Spline')
plt.grid(True)
plt.savefig('/home/conor/crypto/ClampedNatural.png')
plt.close() 

# Print the coefficients for each spline segment
for i in range(len(a_clamped)):
    print(f"Clamped S_{i+1}(x) = {a_clamped[i]:.2f}(x - {x_points[i]})^3 + {b_clamped[i]:.2f}(x - {x_points[i]})^2 + {c_clamped[i]:.2f}(x - {x_points[i]}) + {d_clamped[i]:.2f} for {x_points[i]} <= x < {x_points[i+1]}")

for i in range(len(a_natural)):
    print(f"Natural S_{i+1}(x) = {a_natural[i]:.2f}(x - {x_points[i]})^3 + {b_natural[i]:.2f}(x - {x_points[i]})^2 + {c_natural[i]:.2f}(x - {x_points[i]}) + {d_natural[i]:.2f} for {x_points[i]} <= x < {x_points[i+1]}")
