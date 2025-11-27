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

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a UR3 robot object
ur3 = rtb.models.UR3()

# Initial configuration
ur3.q = [0, -np.pi/3, np.pi/2, 0, -np.pi/2, 0]
# ur3.q = [0, -np.pi/3, np.pi/10, 0, -np.pi/2, 0]

# Add the UR3 to the simulator
env.add(ur3)

# Number of joints in the UR3 which we are controlling
n = 6

# Set the desired end-effector pose
ur3.qd = [0, -np.pi/3, np.pi/10, 0, -np.pi/2, 0]
Hod = ur3.fkine(ur3.qd, end=ur3.ee_links[0])

arrived = False

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
        J = ur3.jacobe(ur3.q)[:3, :]
        Jm = -ur3.jacobm(end=ur3.ee_links[0]).reshape((n,))

        rou = 1e-6  # regulisation
        # Q = J.T @ J + Jm.T @ Jm  + np.eye(n) * rou * rou

        Q = J.T @ J + np.eye(n) * rou * rou

        # Linear component of objective function
        c = -J.T @ v[:3]
        
        # Solve for the joint velocities dq
        qd = qp.solve_qp(P=Q, q=c, solver='quadprog')
        
        # Apply the joint velocities to the UR3
        ur3.qd = qd

        # Step the simulator by 50 ms
        env.step(0.05)

        # Update the manipulability scalar
        M = ur3.manipulability(ur3.q)
        # print("The manipulability measure:", M)

        # Print the Workspace end-effector position
        P = ur3.fkine(ur3.qd, end=ur3.ee_links[0]).t
        print("The position is:", P)

    except KeyboardInterrupt:
        print("Interrupted by user")
        break


