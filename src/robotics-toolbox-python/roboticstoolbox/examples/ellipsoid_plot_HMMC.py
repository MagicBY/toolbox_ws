import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# Load your data from the JSON file
# with open('manipulability_data_LS_Hqp.json', 'r') as f:
with open('manipulability_data_Boyu.json', 'r') as f:
    data = json.load(f)

time_list = data['time_list']
manipulability_list = data['manipulability_list']
jacobian_list = data['jacobian_list']
position_list = data['position_list']  # Load end-effector positions

# Create a figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot the end-effector trajectory
positions = np.array(position_list)
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='End-Effector Trajectory', color='navy')

# Set view angle to face towards the X-Y plane
ax.view_init(elev=90, azim=0)

# Label the start and end positions
ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=50, label='Start Position')
ax.text(positions[0, 0], positions[0, 1], positions[0, 2], ' Start', color='green')

ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=50, label='End Position')
ax.text(positions[-1, 0], positions[-1, 1], positions[-1, 2], ' End', color='red')

# Loop through every 2000th Jacobian matrix to plot the manipulability ellipsoid centered at corresponding position
for i in range(0, len(jacobian_list), 800):  # Pick every 2000th data point
    J = np.array(jacobian_list[i])  # Convert to numpy array
    JJ_T = J @ J.T
    eigenvalues, eigenvectors = np.linalg.eig(JJ_T)

    # Create the ellipsoid
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Scale the ellipsoid down by a factor
    scale_factor = 0.05  # Adjust this value as needed
    ellipsoid = scale_factor * np.sqrt(eigenvalues)[:, np.newaxis] * np.array([x.flatten(), y.flatten(), z.flatten()])
    ellipsoid = eigenvectors @ ellipsoid

    # Center the ellipsoid at the corresponding end-effector position
    ellipsoid[0, :] += positions[i, 0]
    ellipsoid[1, :] += positions[i, 1]
    ellipsoid[2, :] += positions[i, 2]

    # Plot the ellipsoid
    ax.plot_surface(ellipsoid[0, :].reshape(x.shape), 
                    ellipsoid[1, :].reshape(y.shape), 
                    ellipsoid[2, :].reshape(z.shape), 
                    alpha=0.1)

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
# ax.set_title('End-Effector Trajectory and Manipulability Ellipsoids Over Time (HMMC)')
ax.legend()

plt.show()
