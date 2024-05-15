import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

# Given data points
x_points = np.array([0, np.pi/2, np.pi])
y_points = np.array([0, 1, 0])

# Find the Lagrange interpolating polynomial
polynomial = lagrange(x_points, y_points)

# Print the polynomial coefficients
print("Polynomial coefficients:", polynomial.coefficients)

# Define a range of x values for plotting and evaluation
x_values = np.linspace(0, np.pi, 100)
y_values = polynomial(x_values)

# Plot the polynomial
plt.plot(x_values, y_values, label='Interpolating Polynomial')
plt.scatter(x_points, y_points, color='red', label='Data Points')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.title('Interpolating Polynomial')
plt.grid(True)
plt.savefig('/home/conor/crypto/poly.png')
plt.close() 

from numpy.polynomial import Polynomial

# Given data points
x_points_2 = np.array([-2, -1, 0, 1, 2])
y_points_2 = np.array([-4, 1, 1, 2, 10])

# Find the polynomial coefficients using numpy's Polynomial.fit
p4 = Polynomial.fit(x_points_2, y_points_2, 3)

# Print the polynomial coefficients
print("Polynomial P4 coefficients:", p4.convert().coef)

# Define a range of x values for plotting and evaluation
x_values_2 = np.linspace(-2, 1, 100)
y_values_2 = p4(x_values_2)

# Plot the polynomial
plt.plot(x_values_2, y_values_2, label='P4(x)')
plt.scatter(x_points_2, y_points_2, color='red', label='Data Points')
plt.xlabel('x')
plt.ylabel('P4(x)')
plt.legend()
plt.title('4th Degree Polynomial P4(x)')
plt.grid(True)
plt.savefig('/home/conor/crypto/p4.png')
plt.close() 



# Calculate polynomial values at specific x points for the first example
specific_x_values = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
calculated_y_values = polynomial(specific_x_values)
print("Polynomial values at specific x points:", calculated_y_values)

# Calculate polynomial values at specific x points for the second example
specific_x_values_2 = np.array([-2, -1, 0, 1])
calculated_y_values_2 = p4(specific_x_values_2)
print("P4(x) values at specific x points:", calculated_y_values_2)
