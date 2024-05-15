import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, y0, t0, t1, h):
    n = int((t1 - t0) / h)
    t = np.linspace(t0, t1, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y

def dy_dt(t, y):
    return (t - y) / 2

# Initial condition and time interval
y0 = 1
t0 = 0
t1 = 3

# Step sizes
steps = [1, 0.5, 0.25, 0.125]

# Solve and plot for each step size
plt.figure(figsize=(10, 8))
for h in steps:
    t, y = euler_method(dy_dt, y0, t0, t1, h)
    plt.plot(t, y, label=f'h = {h}')

# Plotting details
plt.title("Solution of $y' = (t - y)/2$ using Euler's Method")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
# Save the plot instead of showing it
plt.savefig('/home/conor/crypto/Euler_solution_plot.png')
plt.close()  # Close the plot to free up memory
