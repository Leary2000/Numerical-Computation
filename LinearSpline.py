import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Given data points
x_points = np.array([5, 30, 60, 100])
y_points = np.array([1.226, 2.662, 2.542, 1.201])

# Create the piecewise linear spline
linear_spline = interp1d(x_points, y_points, kind='linear')

# Define a range of x values for plotting and evaluation
x_values = np.linspace(5, 100, 500)
y_values = linear_spline(x_values)

# Plot the piecewise linear spline and the data points
plt.plot(x_values, y_values, label='Piecewise Linear Spline')
plt.scatter(x_points, y_points, color='red', label='Data Points')
plt.xlabel('x')
plt.ylabel('S(x)')
plt.legend()
plt.title('Piecewise Linear Spline')
plt.grid(True)
plt.savefig('/home/conor/crypto/LinearSpline.png')
plt.close() 
