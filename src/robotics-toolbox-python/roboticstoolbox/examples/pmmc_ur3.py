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

# Initialize a list to store end-effector positions
positions = []

# Initialize a counter
i = 0

# Define the cleanup function to save the positions matrix to a file
def save_positions():
    positions_array = np.array(positions)
    np.savetxt("end_effector_positions.txt", positions_array, fmt="%.5f")
    print("End-effector positions have been saved to end_effector_positions.txt")

# Register the cleanup function with atexit
atexit.register(save_positions)

while not arrived:
    try:
        # The pose of the ur3's end-effector
        Hoe = ur3.fkine(ur3.q, end=ur3.ee_links[0])

        # Record the position of the ur3's end-effector
        X = Hoe.t
        # positions.append(X)  # Store the position into the list

        # Transform from the end-effector to desired pose
        Hed = Hoe.inv() * Hod

        # Calculate the required end-effector spatial velocity for the robot to approach the goal.
        v, arrived = rtb.p_servo(Hoe, Hod, 0.01)
        arrived = False

        # Gain term (lambda) for control minimization
        Y = 0.001

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
        
        # ------------------------------------------------------------
        # # Compute the Jacobian matrix
        # J = ur3.jacobe(ur3.q, end=ur3.ee_links[0])

        # # Compute the pseudo-inverse of the Jacobian
        # J_pinv = np.linalg.pinv(J)

        # # Translate the Cartesian velocity to joint space velocity
        # qd = J_pinv @ v

        # ------------------------------------------------------------

        # Apply the joint velocities to the UR3
        ur3.qd = qd

        # Step the simulator by 50 ms
        env.step(0.05)

        # Update the manipulability scalar
        M = ur3.manipulability(ur3.q)
        print("The manipulability measure:", M)

        # Print the Workspace end-effector position
        P = ur3.fkine(ur3.qd, end=ur3.ee_links[0]).t
        # print("The position is:", P)


    except KeyboardInterrupt:
        print("Interrupted by user")
        break

# Save the positions matrix when the program is manually stopped
save_positions()
