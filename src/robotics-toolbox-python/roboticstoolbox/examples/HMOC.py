#!/usr/bin/env python
"""
Manipulability maximization position control for UR3 robot with quaternion regulation
"""
import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
from scipy.optimize import minimize
import json

# Initialize simulator and robot
env = swift.Swift()
env.launch()
ur3 = rtb.models.UR3()
# ur3.q = [0, -np.pi/2, 0, 0, 0, 0] 
ur3.q = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
# ur3.q = [0.7854, -1.0472, 1.3963, -1.5708, 0.8727, 2.0944]
# ur3.q = [0, -0.62, -0.08, 0.23, -0.74, -1.54] # Sim_1
# ur3.q = [0, -1.3963, 0.6981, -0.8727, -1.5708, 6.2832] # H2F

env.add(ur3)

# Robot parameters
n = 6
# ur3.qd = [0, -np.pi/2, np.pi/2, 0, -np.pi/2, 0]
# ur3.qd = [0.9599, -0.7854, 1.2217, -1.7453, 0.6981, 2.2689]
ur3.qd = [0.7854, -1.0472, 1.3963, -1.5708, 0.8727, 2.0944]
# ur3.qd = [0, -1.40, 0.95, 0.64, -0.41, 0.42]

Hod = ur3.fkine(ur3.qd, end=ur3.ee_links[0])

# Hod = sm.SE3(-0.352842, 0.211192, 0.316769)  # H2F

# Simulation setup
qd = np.zeros(n)
arrived = False
dt = 0.05
lambda_penalty = 10
time_elapsed = 0

# Data storage
time_list = []
manipulability_list = []
jacobian_list = []
position_list = []  # EE position (for S1 validation)
velocity_error_list = []  # ||J_t @ qd - v|| (for S1 validation)
theta_e_list = []  # Orientation error angle (for S2 evaluation)

# Joint velocity limits
q_dot_max = np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
q_dot_min = -q_dot_max
A_ineq = np.vstack([np.eye(n), -np.eye(n)])
b_ineq = np.concatenate([q_dot_max, -q_dot_min])

# Quaternion utility functions
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

# Main simulation loop
while not arrived:
    try:
        Hoe = ur3.fkine(ur3.q, end=ur3.ee_links[0])
        X = Hoe.t
        R_ee = Hoe.R

        V, arrived = rtb.p_servo(Hoe, Hod, 0.05)
        arrived = False
        # print(f"Velocity command: {V}")

        # S1: Translational Task
        J_t = ur3.jacobe(ur3.q)[:3, :]
        jacobian_list.append(J_t.tolist())
        v = V[:3]
        
        def objective_s1(q_dot):
            return 0.5 * np.linalg.norm(J_t @ q_dot - v)**2

        result_s1 = minimize(objective_s1, qd)
        q1 = result_s1.x

        # S2: Manipulability and Quaternion Orientation
        Jm = ur3.jacobm(end=ur3.ee_links[0]).reshape((n,))
        I_n = np.eye(n)
        J_r = ur3.jacobe(ur3.q)[3:, :]
        J_t_pinv = np.linalg.pinv(J_t)
        N = I_n - J_t_pinv @ J_t

        q_ee = sm.UnitQuaternion(R_ee).vec
        q_desired = np.array([0.7071, 0, 0.7071, 0])  # 90° around Y-axis

        def objective_s2(q_dot_null):
            q_dot_total = q1 + N @ q_dot_null
            manip_term = 0.5 * np.linalg.norm(I_n @ q_dot_total - Jm)**2
            omega = J_r @ q_dot_total
            q_omega = np.concatenate(([0], omega))
            dq = 0.5 * quat_mult(q_ee, q_omega)
            q_next = q_ee + dt * dq
            q_next /= np.linalg.norm(q_next)
            q_error_next = quat_mult(quat_conj(q_desired), q_next)
            theta_vec = q_error_next[1:]
            penalty_quat = lambda_penalty * np.linalg.norm(theta_vec)**2
            return manip_term + penalty_quat

        def constraint_s2_ineq(q_dot_null):
            q_dot_total = q1 + N @ q_dot_null
            return b_ineq - A_ineq @ q_dot_total

        constraints_s2 = [{'type': 'ineq', 'fun': constraint_s2_ineq}]
        q_dot_null0 = np.zeros(n)
        result_s2 = minimize(objective_s2, q_dot_null0, constraints=constraints_s2)
        if not result_s2.success:
            print(f"S2 optimization failed: {result_s2.message}")

        qd = q1 + N @ result_s2.x

        # Evaluate S1: Compute velocity error ||J_t @ qd - v||
        velocity_error = np.linalg.norm(J_t @ qd - v)
        velocity_error_list.append(velocity_error)

        # Evaluate S2: Compute theta_e (orientation error angle)
        q_error = quat_mult(quat_conj(q_desired), q_ee)  # Current error
        w_e = q_error[0]
        theta_e = 2 * np.arccos(np.clip(w_e, -1.0, 1.0))  # Error angle in radians
        theta_e_list.append(theta_e)

        # Apply joint velocities
        ur3.qd = qd
        env.step(dt)

        # Store data
        M = ur3.manipulability(ur3.q)
        # print(f"Manipulability measure: {M}")
        P = ur3.fkine(ur3.q, end=ur3.ee_links[0]).t
        position_list.append(P.tolist())
        time_elapsed += dt
        time_list.append(time_elapsed)
        manipulability_list.append(M)

    except KeyboardInterrupt:
        print("Interrupted by user")
        break

# Save data to JSON
data_to_save = {
    "time_list": time_list,
    "manipulability_list": manipulability_list,
    "jacobian_list": jacobian_list,
    "position_list": position_list,
    "velocity_error_list": velocity_error_list,
    "theta_e_list": theta_e_list
}
with open('manipulability_data_HMOC.json', 'w') as f:
    json.dump(data_to_save, f)

print("Data saved to manipulability_data_HMOC.json")