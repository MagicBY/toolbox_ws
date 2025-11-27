#!/usr/bin/env python
"""
Implementation of the manipulability maximization position control on UR3 robot
Based on Haviland and Corke's reactive manipulability-maximising controller
"""
import swift
import roboticstoolbox as rtb
import spatialmath as sm
from spatialmath import SE3
import numpy as np
import qpsolvers as qp
import json

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a UR3 robot object
ur3 = rtb.models.UR3()

# Initial configuration
ur3.q = [0, -0.62, -0.08, 0.23, -0.74, -1.54]

# Add the UR3 to the simulator
env.add(ur3)

# Number of joints in the UR3 which we are controlling
n = 6

# Set the desired end-effector pose
ur3.qd = [0, -1.40, 0.95, 0.64, -0.41, 0.42]
Tep = ur3.fkine(ur3.qd, end=ur3.ee_links[0])

arrived = False

# Initialize lists for data storage
time_list = []
manipulability_list = []
jacobian_list = []
position_list = []
time_elapsed = 0

while not arrived:
    try:
        # Current end-effector pose
        Te = ur3.fkine(ur3.q, end=ur3.ee_links[0])

        # Record position
        X = Te.t

        # Transform from current to desired pose
        eTep = Te.inv() * Tep

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

        # Calculate required end-effector spatial velocity
        v, arrived = rtb.p_servo(Te, Tep, 1.0)

        # Gain term for control minimisation
        Y = 0.01

        # Quadratic component of objective function
        Q = np.eye(n + 6)
        Q[:n, :n] *= Y
        Q[n:, n:] = (1 / e) * np.eye(6)

        # Equality constraints
        Aeq = np.c_[ur3.jacobe(ur3.q), np.eye(6)]
        beq = v.reshape((6,))
        jacobian_list.append(ur3.jacobe(ur3.q).tolist())  # Save Jacobian

        # Inequality constraints for joint limit avoidance
        Ain = np.zeros((n + 6, n + 6))
        bin = np.zeros(n + 6)

        # Joint limit parameters
        ps = 0.05  # Minimum angle to joint limit
        pi = 0.9   # Influence angle for velocity damper

        # Form joint limit velocity damper
        Ain[:n, :n], bin[:n] = ur3.joint_velocity_damper(ps, pi, n)

        # Linear component: manipulability Jacobian
        c = np.r_[-ur3.jacobm(end=ur3.ee_links[0]).reshape((n,)), np.zeros(6)]

        # Solve for joint velocities
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, solver='quadprog')

        # Apply joint velocities
        ur3.qd[:n] = qd[:n]

        # Step simulator
        env.step(0.05)

        # Calculate and store data
        M = ur3.manipulability(ur3.q)
        P = ur3.fkine(ur3.q, end=ur3.ee_links[0]).t
        
        time_elapsed += 0.05
        time_list.append(time_elapsed)
        manipulability_list.append(M)
        position_list.append(P.tolist())
        
        print(f"Manipulability: {M:.4f}, Position: {P}")

    except KeyboardInterrupt:
        print("Interrupted by user")
        break

# Save data to JSON file
data_to_save = {
    "time_list": time_list,
    "manipulability_list": manipulability_list,
    "jacobian_list": jacobian_list,
    "position_list": position_list,
}

with open('manipulability_data_Haviland.json', 'w') as f:
    json.dump(data_to_save, f)

print("Data saved to manipulability_data_Haviland.json")