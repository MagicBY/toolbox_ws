#!/usr/bin/env python
"""
Implementation of the manipulability maximization position control on UR3 robot (Used in confirmation report)
"""
import swift
import roboticstoolbox as rtb
import spatialmath as sm
from spatialmath import SE3
import numpy as np
import qpsolvers as qp
import atexit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a UR3 robot object
ur3 = rtb.models.UR3()

# Initial configuration
# ur3.q = [0.7854, -1.0472, 1.3963, -1.5708, 0.8727, 2.0944]
ur3.q = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
# ur3.q = [0, -0.62, -0.08, 0.23, -0.74, -1.54]

# Add the UR3 to the simulator
env.add(ur3)

# Number of joints in the UR3 which we are controlling
n = 6

# Set the desired end-effector pose
# ur3.qd = [0, -1.40, 0.95, 0.64, -0.41, 0.42]
ur3.qd = [0, -np.pi/2, np.pi/2, 0, -np.pi/2, 0]
Hod = ur3.fkine(ur3.qd, end=ur3.ee_links[0])

arrived = False

# Initialization
qd = np.zeros((6,))  

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
        V, arrived = rtb.p_servo(Hoe, Hod, 1.0)
        arrived = False

        # ---------------------------------------------------------------------

        # *****Primary objective function
        J = ur3.jacobe(ur3.q)[:3, :]
        jacobian_list.append(J.tolist())  # Save the Jacobian matrix
        v = V[:3]

        # Objective function
        def objective(q_dot):
            return np.linalg.norm(J @ q_dot - v)**2
        
        # Initial guess for q_dot
        q_dot0 = qd

        # Perform the optimization
        q1 = minimize(objective, q_dot0)

        # *****Secondary objective function
        Jm = -ur3.jacobm(end=ur3.ee_links[0]).reshape((n,))

        Q = np.eye(n)
        c = -Q @ Jm.T

        def objective(q_dot):
            return np.linalg.norm(Q @ q_dot - c)
        
        def constraint(q_dot):
            return J @ q_dot - J @ q1.x
        
        # Define the constraint in the form expected by scipy.optimize
        constraints = {'type': 'eq', 'fun': constraint}

        q2 = minimize(objective, q_dot0, constraints=constraints)

        qd = q2.x  

        # ------------------------------------------------------------
        # Compared with no special effect
        # # Compute the Jacobian matrix
        # J = ur3.jacobe(ur3.q, end=ur3.ee_links[0])

        # # Compute the pseudo-inverse of the Jacobian
        # J_pinv = np.linalg.pinv(J)

        # # Translate the Cartesian velocity to joint space velocity
        # qd = J_pinv @ V

        # print(f"Velocity command: {V}")

        # ------------------------------------------------------------
        
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

with open('manipulability_data_Boyu.json', 'w') as f:
    json.dump(data_to_save, f)

print("Data saved to manipulability_data_Boyu.json")

