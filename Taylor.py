import sympy as sp
import numpy

def taylor_series(function, a, n):
    """
    Compute the Taylor series approximation of the given function at point a up to the n-th derivative.

    Parameters:
    - function: sympy function, the function to approximate.
    - a: float or int, the point around which to approximate the function.
    - n: int, the degree of the Taylor polynomial.

    Returns:
    - sympy expression, the Taylor series approximation of the function.
    """
    x = sp.symbols('x')  # Define the symbol
    taylor_series = 0

    for i in range(n + 1):
        # Compute the i-th derivative
        derivative = sp.diff(function, x, i)
        # Evaluate the derivative at point a
        derivative_at_a = derivative.subs(x, a)
        # Add the term to the series
        taylor_series += (derivative_at_a * (x - a)**i) / sp.factorial(i)

    return taylor_series


def F(x):
    return sp.log(1 + x)

# Example usage
if __name__ == "__main__":
    x = sp.symbols('x')
    # Define the function you want to approximate
    function = F(x)
    # Point around which to approximate
    a = 0
    # Initially, let's not specify n; we'll determine it based on accuracy requirements
    # n = 5  # This was arbitrary; let's find the required n for the desired accuracy

    # Instead of directly choosing n, let's find the minimum n required for the accuracy
    for n in range(1, 10):  # Start with 1 and incrementally test higher degrees
        approximation = taylor_series(function, a, n)
        # Test the approximation at the endpoints of the interval [-0.5, 0.5]
        error_at_minus_half = abs(approximation.subs(x, -0.5) - function.subs(x, -0.5))
        error_at_half = abs(approximation.subs(x, 0.5) - function.subs(x, 0.5))
        if max(error_at_minus_half, error_at_half) < 10**-3:
            print(f"Minimum n required for accuracy within 10^-3: {n}")
            print(f"Taylor series approximation up to the {n}-th term: {approximation}")
            break