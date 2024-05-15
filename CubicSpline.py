import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Given data points
velocity = np.array([0, 50, 75, 100])
drag_coefficient = np.array([0.5, 0.5, 0.4, 0.28])

# Create a cubic spline interpolation model
cs = CubicSpline(velocity, drag_coefficient)

# Velocity of interest
v_target = 95

# Estimate the drag coefficient at the target velocity
c_v_target = cs(v_target)

print(f"Estimated drag coefficient for a velocity of {v_target} mph is {c_v_target:.3f}")

# Optional: Plot the spline and data for visual inspection
velocities = np.linspace(0, 100, 500)
plt.plot(velocities, cs(velocities), label='Cubic Spline')
plt.scatter(velocity, drag_coefficient, color='red', label='Data Points')
plt.title('Drag Coefficient vs. Velocity')
plt.xlabel('Velocity (mph)')
plt.ylabel('Drag Coefficient (C(v))')
plt.legend()
plt.grid(True)
plt.savefig('/home/conor/crypto/CubicSpline.png')
plt.close() 