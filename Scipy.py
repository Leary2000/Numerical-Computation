from scipy.optimize import root_scalar
from scipy.optimize import minimize
import math

from scipy import optimize

def f(x):
    return x**3 - 1  # Has a real root at x = 1

def fprime(x):
    return 3*x**2

def f_p_pp(x):
    return (x**3 - 1), 3*x**2, 6*x

#Distance for large numbers
def safe_euclidean_distance(x, y):
    if x == 0 and y == 0:
        return 0
    elif abs(x) > abs(y):
        return abs(x) * math.sqrt(1 + (y/x)**2)
    else:
        return abs(y) * math.sqrt(1 + (x/y)**2)


def decimal_to_binary_mantissa(decimal_number):
    """Convert a decimal number to its binary mantissa representation."""
    if decimal_number == 0:
        return '0'
    
    # Normalize the number (between 1 and 2 for the mantissa extraction)
    while decimal_number >= 2:
        decimal_number /= 2
    while decimal_number < 1:
        decimal_number *= 2
    
    # Remove the leading '1.' for the mantissa
    decimal_number -= 1
    
    # Convert the fractional part to binary
    binary_mantissa = ''
    for _ in range(52):  # Adjust for desired precision
        decimal_number *= 2
        if decimal_number >= 1:
            binary_mantissa += '1'
            decimal_number -= 1
        else:
            binary_mantissa += '0'
    
    return binary_mantissa


# Using Brent's method with a bracket
sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')
print(sol.root, sol.iterations, sol.function_calls)


# Using Newton's method with value and derivatives in a single call
sol = optimize.root_scalar(f_p_pp, x0=0.2, fprime=True, method='newton')
print(sol.root, sol.iterations, sol.function_calls)

# Using Halley's method with value, first and second derivatives
sol = optimize.root_scalar(f_p_pp, x0=0.2, fprime=True, fprime2=True, method='halley')
print(sol.root, sol.iterations, sol.function_calls)


sol = optimize.root_scalar(f, x0=0.2, fprime=fprime, method='newton')
print(sol.root, sol.iterations, sol.function_calls)




# Example usage
x = 2e+100
y = 4e+100
distance = safe_euclidean_distance(x, y)

print(distance)

print(decimal_to_binary_mantissa(25))
