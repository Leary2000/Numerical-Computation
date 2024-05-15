import numpy as np

def second_derivative_2nd_order(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)

def second_derivative_4th_order(f, x, h):
    return (-f(x + 2*h) + 16 * f(x + h) - 30 * f(x) + 16 * f(x - h) - f(x - 2*h)) / (12 * h**2)

# Define the function
def f(x):
    return np.cos(x)

# Point of interest and step sizes
x = 0.8
h_values = [0.1, 0.01, 0.001, 0.0001,0.00001,0.000001, 0.0000001]

# Calculations
results = {}
for h in h_values:
    derivative_2nd_order = second_derivative_2nd_order(f, x, h)
    derivative_4th_order = second_derivative_4th_order(f, x, h)
    results[h] = (derivative_2nd_order, derivative_4th_order)

# Display results
for h in h_values:
    print(f"For h = {h}:")
    print(f"  2nd Order Approximation: {results[h][0]}")
    print(f"  4th Order Approximation: {results[h][1]}")
