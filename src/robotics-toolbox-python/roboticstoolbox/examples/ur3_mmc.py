#!/usr/bin/env python
"""
Implementation of the manipulability maximization control on UR3 robot
"""
import swift
import roboticstoolbox as rtb
import spatialmath as sm
from spatialmath import SE3
import numpy as np
import qpsolvers as qp

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

# Set the desired end-effector pose
ur3.qd = [0, -np.pi/3, np.pi/10, 0, -np.pi/2, 0]
Hod = ur3.fkine(ur3.qd, end=ur3.ee_links[0])

arrived = False

while not arrived:
    # The pose of the ur3's end-effector
    Hoe = ur3.fkine(ur3.q, end=ur3.ee_links[0])

    # Print the end-effector orientation
    Ang = Hoe.eul()
    print('Orientation: ', Ang)

    # Transform from the end-effector to desired pose
    Hed = Hoe.inv() * Hod

    # Calculate the required end-effector spatial velocity for the robot to approach the goal.
    v, arrived = rtb.p_servo(Hoe, Hod, 1)
    # print(v)

    # Gain term (lambda) for position servo control minimization
    Y = 0.05
    arrived = False

    # Quadratic component of objective function
    Q = np.eye(n)

    # Joint velocity component of Q
    Q[:n, :n] *= Y
    
    # Equality constraints
    Aeq = np.zeros((6,n))
    Aeq[:3, :n] = ur3.jacobe(ur3.q)[:3, :]

    beq = np.zeros(6)
    beq[:3] = v[:3]

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
    c = -ur3.jacobm(end=ur3.ee_links[0]).reshape((n,))
    # c = np.zeros(6,)  # Compare to no manipulability maximization

    # Solve for the joint velocities dq
    qd = qp.solve_qp(P=Q, q=c, G=Ain, h=bin, A=Aeq, b=beq, solver='quadprog')
    
    # Apply the joint velocities to the UR3
    ur3.qd[:n] = qd[:n]

    # Step the simulator by 50 ms
    env.step(0.05)

    # Update the manipulability scalar
    M = ur3.manipulability(ur3.q)
    # print("The manipulability measure is:", M)

    

    

