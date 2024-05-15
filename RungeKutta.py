import numpy as np
import matplotlib.pyplot as plt

def runge_kutta(f, x0, y0, xf, h):
    x = np.arange(x0, xf + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    for i in range(1, len(x)):
        k1 = h * f(x[i-1], y[i-1])
        k2 = h * f(x[i-1] + 0.5 * h, y[i-1] + 0.5 * k1)
        k3 = h * f(x[i-1] + 0.5 * h, y[i-1] + 0.5 * k2)
        k4 = h * f(x[i-1] + h, y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x, y

def dy_dx(x, y):
    return (x - y) / 2

# Initial conditions
x0, y0 = 0, 1
xf = 3

# Step sizes
steps = [1, 0.5, 0.25, 0.125]

# Plotting the results
plt.figure(figsize=(10, 8))
for h in steps:
    x, y = runge_kutta(dy_dx, x0, y0, xf, h)
    plt.plot(x, y, label=f'h = {h}')

plt.title("Solution of $y' = (x - y)/2$ using RK4")
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.grid(True)
plt.savefig('/home/conor/crypto/RK_solution_plot.png')
plt.close()  # Close the plot to free up memory
