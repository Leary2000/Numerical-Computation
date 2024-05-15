import numpy as np

# Points data
points = np.array([(-2, -2), (-1, 1), (2, -1)])

# Constructing Vandermonde matrix
x_values = points[:, 0]
f_values = points[:, 1]
degree = len(points) - 1  # Degree of the polynomial
A = np.vander(x_values, degree + 1)

# Solve for coefficients
coefficients = np.linalg.solve(A, f_values)

print("Coefficients of the interpolating polynomial:", coefficients)
import matplotlib.pyplot as plt

# Function to evaluate polynomial at x
def eval_poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(reversed(coeffs)))

# Plotting the points
plt.scatter(x_values, f_values, color='red', label='Data points')

# Plotting the polynomial
x_plot = np.linspace(min(x_values) - 1, max(x_values) + 1, 400)
y_plot = [eval_poly(coefficients, x) for x in x_plot]
plt.plot(x_plot, y_plot, label='Interpolating Polynomial')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.title('Interpolating Polynomial through Given Points')
plt.savefig('/home/conor/crypto/Vandermonde.png')
plt.close() 
