#!/usr/bin/env python
"""
Script to plot UR3 controller evaluation: trajectory, orientation error, and relative trajectory error over time
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import roboticstoolbox as rtb
import spatialmath as sm

# Load data from JSON
with open('manipulability_data_HMOC.json', 'r') as f:
    data = json.load(f)

time_list = np.array(data["time_list"])
position_list = np.array(data["position_list"])
theta_e_list = np.array(data["theta_e_list"])

# Initial and goal positions
initial_q = [0, -0.62, -0.08, 0.23, -0.74, -1.54]
goal_q = [0, -1.40, 0.95, 0.64, -0.41, 0.42]

ur3 = rtb.models.UR3()
initial_pose = ur3.fkine(initial_q, end=ur3.ee_links[0])
# goal_pose = ur3.fkine(goal_q, end=ur3.ee_links[0])
goal_pose = sm.SE3(-0.352842, 0.211192, 0.316769)  # H2F

p_initial = initial_pose.t  # Initial EE position [x, y, z]
p_goal = goal_pose.t       # Goal EE position [x, y, z]

# Verify and print path length
total_path_length = np.linalg.norm(p_goal - p_initial)
print(f"Initial Position: {p_initial}")
print(f"Goal Position: {p_goal}")
print(f"Total Path Length: {total_path_length:.3f} m")

# Desired straight-line trajectory
num_points = len(time_list)
desired_trajectory = np.linspace(p_initial, p_goal, num_points)  # Linear interpolation over time

# Actual trajectory from position_list
actual_trajectory = position_list

# Compute absolute trajectory error (Euclidean distance)
absolute_trajectory_error = np.linalg.norm(actual_trajectory - desired_trajectory, axis=1)

# Compute component-wise errors for debugging
x_error = np.abs(actual_trajectory[:, 0] - desired_trajectory[:, 0])
y_error = np.abs(actual_trajectory[:, 1] - desired_trajectory[:, 1])
z_error = np.abs(actual_trajectory[:, 2] - desired_trajectory[:, 2])
print(f"Max X Error: {max(x_error):.3f} m")
print(f"Max Y Error: {max(y_error):.3f} m")
print(f"Max Z Error: {max(z_error):.3f} m")

# Compute relative trajectory error (as percentage of total path length, capped at 100%)
relative_trajectory_error = (absolute_trajectory_error / total_path_length) * 100
relative_trajectory_error = np.clip(relative_trajectory_error, 0, 100)  # Cap at 100% for sanity

# # Plot 1: Desired vs. Actual Trajectory (3D)
# fig1 = plt.figure(figsize=(10, 8))
# ax1 = fig1.add_subplot(111, projection='3d')

# # Desired trajectory
# ax1.plot(desired_trajectory[:, 0], desired_trajectory[:, 1], desired_trajectory[:, 2], 
#          'b--', label='Desired Trajectory', linewidth=2)

# # Actual trajectory
# ax1.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], actual_trajectory[:, 2], 
#          'r-', label='Actual Trajectory', linewidth=2)

# # Initial and goal points
# ax1.scatter([p_initial[0]], [p_initial[1]], [p_initial[2]], c='g', marker='o', label='Initial Position')
# ax1.scatter([p_goal[0]], [p_goal[1]], [p_goal[2]], c='m', marker='x', label='Goal Position')

# ax1.set_xlabel('X (m)')
# ax1.set_ylabel('Y (m)')
# ax1.set_zlabel('Z (m)')
# ax1.set_title('Desired vs. Actual EE Trajectory')
# ax1.legend()
# ax1.grid(True)

# Plot 2: Theta_e Over Time
fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111)

ax2.plot(time_list, theta_e_list, 'b-', label='Orientation Error (θ_e)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('θ_e (rad)')
ax2.set_title('Orientation Error Over Time')
ax2.grid(True)
ax2.legend()

# # Plot 3: Relative Trajectory Error Over Time (Vertical Bars Every 10 Timesteps)
# fig3 = plt.figure(figsize=(10, 6))
# ax3 = fig3.add_subplot(111)

# # Select every 10th timestep for bars
# sampling_interval = 10
# sampled_times = time_list[::sampling_interval]
# sampled_errors = relative_trajectory_error[::sampling_interval]

# # Vertical bars (bar chart) for relative error
# bar_width = 0.1  # Wider bars for visibility with fewer points, adjust as needed
# ax3.bar(sampled_times, sampled_errors, width=bar_width, color='r', 
#         label=f'Relative Trajectory Error (Total Path = {total_path_length:.3f} m)')

# ax3.set_xlabel('Time (s)')
# ax3.set_ylabel('Relative Error (%)')
# ax3.set_title('Relative Trajectory Error Over Time (Every 10th Timestep)')
# ax3.grid(True)
# ax3.legend()

# # Adjust y-axis limits for better visibility (optional, based on your data)
# ax3.set_ylim(0, max(sampled_errors) * 1.2)  # Extend slightly above max for clarity

# Show plots
plt.show()