import matplotlib.pyplot as plt
import numpy as np

# Create x values
x = np.linspace(0, 10, 500)

# Create 3 y values
y1 = np.sin(x)
y2 = np.sin(x + 1)
y3 = np.sin(x + 2)

# Plot them
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='sin(x + 1)')
plt.plot(x, y3, label='sin(x + 2)')

# Add labels and title
plt.title("Test Plot with Custom matplotlibrc")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# Add grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.show()
