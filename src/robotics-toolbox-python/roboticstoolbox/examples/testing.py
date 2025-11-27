#!/usr/bin/env python
"""
Script to confirm UR3 EE pointing direction in Swift by printing initial X, Y, Z axes
"""
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a UR3 robot object
ur3 = rtb.models.UR3()

# Initial configuration (EE pointing downward)
ur3.q = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi]

# Add the UR3 to the simulator
env.add(ur3)

# Get the EE pose
Hoe = ur3.fkine(ur3.q, end=ur3.ee_links[0])
R_ee = Hoe.R  # Rotation matrix of EE frame in base frame

# Print the EE X, Y, Z axes
print(f"EE X-axis: {R_ee[:, 0]}")
print(f"EE Y-axis: {R_ee[:, 1]}")
print(f"EE Z-axis: {R_ee[:, 2]}")

# Hold the simulator open to observe
env.hold()