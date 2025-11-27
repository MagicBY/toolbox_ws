#!/usr/bin/env python
"""
Implementation of the manipulability maximization position control on UR3 robot
"""
import swift
import roboticstoolbox as rtb
import spatialmath as sm
from spatialmath import SE3
import numpy as np
import qpsolvers as qp
import atexit
import matplotlib.pyplot as plt
import json


# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a UR3 robot object
ur3 = rtb.models.UR3()

# Initial configuration
# ur3.q = [0, -np.pi/2, -np.pi/6, -np.pi/6, -np.pi/6, 0]
# ur3.q = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
ur3.q = [0, -0.62, -0.08, 0.23, -0.74, -1.54]

# Add the UR3 to the simulator
env.add(ur3)

# Number of joints in the UR3 which we are controlling
n = 6

# Set the desired end-effector pose
# ur3.qd = [0, -np.pi/2, 0, 0, 0, 0]
# ur3.qd = [0.9599, -0.7854, 1.2217, -1.7453, 0.6981, 2.2689]
ur3.qd = [0, -1.40, 0.95, 0.64, -0.41, 0.42]
Hod = ur3.fkine(ur3.qd, end=ur3.ee_links[0])

arrived = False

# Initialize lists to store time and manipulability measure
time_list = []
manipulability_list = []
jacobian_list = []  
position_list = []

time_elapsed = 0

while not arrived:
    try:
        # The pose of the ur3's end-effector
        Hoe = ur3.fkine(ur3.q, end=ur3.ee_links[0])

        # Record the position of the ur3's end-effector
        X = Hoe.t

        # Transform from the end-effector to desired pose
        Hed = Hoe.inv() * Hod

        # Calculate the required end-effector spatial velocity for the robot to approach the goal.
        v, arrived = rtb.p_servo(Hoe, Hod, 0.01)
        arrived = False

        # Quadratic component of objective function
        Jm = ur3.jacobm(end=ur3.ee_links[0]).reshape((n,))

        rou = 1e-6  # regulisation
        Q = Jm.T @ Jm  + np.eye(n) * rou * rou

        # Linear component of objective function
        c = np.zeros(6,)

        # Equality constraints
        Aeq = np.zeros((6, n))
        J = ur3.jacobe(ur3.q)[:3, :]
        jacobian_list.append(J.tolist())  # Save the Jacobian matrix
        Aeq[:3, :n] = J

        beq = np.zeros(6)
        beq[:3] = v[:3]
        
        # Solve for the joint velocities dq
        qd = qp.solve_qp(P=Q, q=c, A=Aeq, b=beq, solver='quadprog')
        
        # Apply the joint velocities to the UR3
        ur3.qd = qd

        # Step the simulator by 50 ms
        env.step(0.05)

        # Update the manipulability scalar
        M = ur3.manipulability(ur3.q)
        print("The manipulability measure:", M)

        # Print the Workspace end-effector position
        P = ur3.fkine(ur3.q, end=ur3.ee_links[0]).t
        # print("The position is:", P)
        position_list.append(P.tolist())  # Save the Jacobian matrix

        time_elapsed += 0.05
        time_list.append(time_elapsed)
        manipulability_list.append(M)

    except KeyboardInterrupt:
        print("Interrupted by user")
        break

# Save time_list and manipulability_list to a JSON file
data_to_save = {
    "time_list": time_list,
    "manipulability_list": manipulability_list,
    "jacobian_list": jacobian_list, 
    "position_list": position_list,
}

with open('manipulability_data_Haviland.json', 'w') as f:
    json.dump(data_to_save, f)

print("Data saved to manipulability_data_Haviland.json")

