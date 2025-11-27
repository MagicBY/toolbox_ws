import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Load the data from the JSON file
with open('manipulability_data_Haviland.json', 'r') as f:
    data = json.load(f)

time_list = data['time_list']
jacobian_list = data['jacobian_list']

# Function to compute and return the sizes and rotation angle of the ellipsoid axes
def compute_ellipsoid_properties(J, proj_axis):
    J_proj = J[proj_axis, :]
    U, S, Vt = np.linalg.svd(J_proj)
    sizes = S  # Use the singular values directly
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))  # Calculate rotation angle in degrees
    return sizes, angle

# Extract sizes and rotation angles for every 600th ellipsoid and compute condition numbers
ellipsoid_properties = []

for i in range(0, len(jacobian_list), 500):
    J = np.array(jacobian_list[i])
    properties = {
        'yz': compute_ellipsoid_properties(J, [1, 2]),
        'xz': compute_ellipsoid_properties(J, [0, 2]),
        'xy': compute_ellipsoid_properties(J, [0, 1])
    }
    ellipsoid_properties.append(properties)

# Function to plot the ellipsoids
def plot_ellipsoids(ax, properties, color, alpha, x_offset, scale_factor):
    for i, (size, angle) in enumerate(properties):
        width = size[0] * scale_factor
        height = size[1] * scale_factor
        # Shift the ellipses so that the center of the first ellipse is at t=2.0
        ellipse = Ellipse((x_offset + i * 2.5 + 2.0, 0), width, height, angle=angle, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(ellipse)
    ax.set_aspect('equal')

# Define a scaling factor
scale_factor = 10  # Adjust this factor to scale the ellipses

# Create subplots for the three projections and the condition number plot (arranged vertically)
fig, axes = plt.subplots(3, 1, figsize=(10, 15))  

# Define colors for the plots
colors = ['blue', 'green', 'red']

# Plot the ellipsoids for each projection
projections = ['yz', 'xz', 'xy']
for idx, (proj, ax, color) in enumerate(zip(projections, axes[:3], colors)):  # Only plot on the first three axes
    properties = [prop[proj] for prop in ellipsoid_properties]
    plot_ellipsoids(ax, properties, color=color, alpha=0.3, x_offset=0, scale_factor=scale_factor)
    
    # Set titles and labels
    if proj == 'yz':
        ax.set_ylabel('M_yz')
        ax.set_xlabel('t')
    elif proj == 'xz':
        ax.set_ylabel('M_xz')
        ax.set_xlabel('t')
    else:
        ax.set_ylabel('M_xy')
        ax.set_xlabel('t')

    # Set limits and ticks
    x_max = len(ellipsoid_properties) * 2.5 + 3.0
    y_max = 5
    ax.set_xlim(0, x_max)
    ax.set_ylim(-y_max, y_max)

    # Remove horizontal axis ticks and set vertical axis ticks at -5 and 5
    ax.set_xticks([])  # No ticks on the horizontal axis
    ax.set_yticks([-5.0, 5.0])  

# Show the plot
plt.tight_layout()
plt.show()
