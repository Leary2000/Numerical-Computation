import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the ODEs and initial conditions as before
def exercise1(t, y):
    return [y[1], -4*t/9]

def exercise2(t, y):
    return [-y/t]

def exercise3(t, y):
    return [-y/t]

sol1 = solve_ivp(exercise1, [0, 2], [2, 0], dense_output=True)
sol2 = solve_ivp(exercise2, [1, 3], [1], dense_output=True)
sol3 = solve_ivp(exercise3, [0.1, 2], [-2], dense_output=True)  # Start from 0.1 to avoid division by zero

# Generate a range of t values for plotting
t_points = np.linspace(0, 2, 100)
t_points3 = np.linspace(0.1, 2, 100)

# Create the plot
plt.figure(figsize=(12, 8))

plt.subplot(311)
plt.plot(t_points, sol1.sol(t_points)[0], label='9y\'\' + 4x = 0, y(0) = 2')
plt.legend()

plt.subplot(312)
plt.plot(t_points, sol2.sol(t_points)[0], label="y' = -y/x, y(1) = 1")
plt.legend()

plt.subplot(313)
plt.plot(t_points3, sol3.sol(t_points3)[0], label="ty' + y = 0, y(2) = -2")
plt.legend()

# Save the plot instead of showing it
plt.savefig('/home/conor/crypto/IVP_solution_plot.png')
plt.close()  # Close the plot to free up memory
