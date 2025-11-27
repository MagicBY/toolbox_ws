#!/usr/bin/env python
"""
Plot the end-effector positions from the saved text file.
"""
import numpy as np
import matplotlib.pyplot as plt

# Load the positions from the text file
positions = np.loadtxt("end_effector_positions.txt")

# Extract the x, y, and z coordinates
x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the positions and connect them with lines
ax.plot(x, y, z, marker='o', linestyle='-', markersize=2, linewidth=1)

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('End-Effector Positions')

# Show the plot
plt.show()
