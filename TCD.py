import numpy as np

def derivative(f, x, h):
    return (f(x + h) - f(x)) / h

def find_optimal_h(f, x, initial_h=1.0, tolerance=1e-6):
    previous_derivative = derivative(f, x, initial_h)
    h = initial_h / 2
    current_derivative = derivative(f, x, h)
    
    while np.abs(current_derivative - previous_derivative) > tolerance:
        previous_derivative = current_derivative
        h /= 2
        current_derivative = derivative(f, x, h)
        
        # Check if changes start increasing, which may indicate optimal h found
        if np.abs(current_derivative - previous_derivative) < tolerance:
            return h * 2  # Return the previous h before error increased

    return h

# function to differentiate
def f(x):
    return np.exp(x)

# Use the function to find the optimal h
x = 1
optimal_h = find_optimal_h(f, x)
print(f"Optimal h is approximately: {optimal_h}")

# Calculate the derivative with the optimal h
approx_derivative = derivative(f, x, optimal_h)
print(f"Approximate derivative at x = {x} is: {approx_derivative}")
print(f"Exact derivative at x = {x} is: {np.exp(1)}")



import numpy as np

def central_difference_2nd_order(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def central_difference_4th_order(f, x, h):
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)

# Define the function
def f(x):
    return np.cos(x)

# Point of interest and step sizes
x = 0.8
h_values = [0.1, 0.01, 0.001, 0.0001]

# Calculations
results = {}
for h in h_values:
    derivative_2nd = central_difference_2nd_order(f, x, h)
    derivative_4th = central_difference_4th_order(f, x, h)
    results[h] = (derivative_2nd, derivative_4th)

# Display results
for h in h_values:
    print(f"For h = {h}:")
    print(f"  2nd Order Approximation: {results[h][0]}")
    print(f"  4th Order Approximation: {results[h][1]}")



# Given distances at each time point
D = {8: 17.453, 9: 21.460, 10: 25.753, 11: 30.301, 12: 35.084}

# Central difference to estimate the derivative at t = 10
velocity_at_10 = (D[11] - D[9]) / 2

print(f"Estimated velocity at t = 10 seconds is: {velocity_at_10} units per second")

# Calculate the exact velocity at t = 10
exact_velocity_at_10 = 7 - 7 / np.exp(1)

print(f"Exact velocity at t = 10 seconds is: {exact_velocity_at_10} units per second")
