import matplotlib.pyplot as plt
import numpy as np

# Set the number of rings
num_rings = 5

# Create a figure and axis
fig, ax = plt.subplots()

# Draw each ring
for i in range(num_rings, 0, -1):
    circle = plt.Circle((0, 0), i, fill=False, linewidth=2, edgecolor='black')
    ax.add_artist(circle)

# Set limits and aspect
ax.set_xlim(-num_rings, num_rings)
ax.set_ylim(-num_rings, num_rings)
ax.set_aspect('equal', adjustable='box')

# Remove axis labels
ax.axis('off')

# Show the plot
plt.show()
