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
from spatialmath import Quaternion

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a UR3 robot object
ur3 = rtb.models.UR3()

# Initial configuration
ur3.q = [0, -np.pi/3, np.pi/10, 0, -np.pi/2, 0]

# Add the UR3 to the simulator
env.add(ur3)

# Number of joints in the UR3 which we are controlling
n = 6

# Set the desired end-effector position and orientation
ur3.qd = [0, -np.pi/3, np.pi/10, 0, -np.pi/2, 0]
Hod = ur3.fkine(ur3.qd, end=ur3.ee_links[0])
Xd = Hod.t
Rod = Hod.R

arrived = False

while not arrived:
    # The position and orientation of the end-effector
    Hoe = ur3.fkine(ur3.q, end=ur3.ee_links[0])
    X = Hoe.t
    Roe = Hoe.R
    Roe_inv = np.linalg.inv(Roe)
    Red = Roe_inv * Rod

    # Combine position and orientation error
    Xe = np.hstack((X-Xd, Red)) #-------------------------

    # Gain term (lambda) for control minimization
    Y = 0.05
    lambda_matrix = np.eye(6)
    lambda_matrix *= Y

    # Quadratic component of objective function
    J = ur3.jacobe(ur3.q, end=ur3.ee_links[0])
    Q = J.T @ J

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n, n))
    bin = np.zeros(n)

    # The minimum angle (in radians) in which the joint is allowed to approach to its limit
    ps = 0.5

    # The influence angle (in radians) in which the velocity damper becomes active
    pi = 1.0

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = ur3.joint_velocity_damper(ps, pi, n)

    # Linear component of objective function: the manipulability Jacobian
    Jm = ur3.jacobm(end=ur3.ee_links[0]).reshape((n, 1))
    c = (J.T @ lambda_matrix @ Xe).reshape((n, 1)) - Jm

    # Solve for the joint velocities dq
    qd = qp.solve_qp(P=Q, q=c, G=Ain, h=bin, solver='quadprog')

    # Apply the joint velocities to the UR3
    ur3.qd = qd

    # Step the simulator by 50 ms
    env.step(0.05)

    # Update the manipulability scalar
    M = ur3.manipulability(ur3.q)
    print("The manipulability measure is:", M)
