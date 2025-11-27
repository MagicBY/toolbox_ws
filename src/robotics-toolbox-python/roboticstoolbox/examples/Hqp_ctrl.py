#!/usr/bin/env python
"""
Implementation of manipulability maximization position control on UR3 robot
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
ur3.q = [0, -0.62, -0.08, 0.23, -0.74, -1.54]
env.add(ur3)

# Robot parameters
n = 6
ur3.qd = [0, -1.40, 0.95, 0.95, -0.41, 0.42]
Hod = ur3.fkine(ur3.qd, end=ur3.ee_links[0])

# Simulation setup
qd = np.zeros(n)
arrived = False
dt = 0.05
lambda_penalty = 1.0
time_elapsed = 0

# Data storage
time_list = []
manipulability_list = []
jacobian_list = []
position_list = []

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

        V, arrived = rtb.p_servo(Hoe, Hod, 0.01)
        arrived = False

        # S1: Translational Task
        J_t = ur3.jacobe(ur3.q)[:3, :]
        jacobian_list.append(J_t.tolist())
        v = V[:3]

        def objective_s1(q_dot):
            return 0.5 * np.linalg.norm(J_t @ q_dot - v)**2

        result_s1 = minimize(objective_s1, qd)
        q1 = result_s1.x

        # S2: Manipulability and Orientation
        Jm = ur3.jacobm(end=ur3.ee_links[0]).reshape((n,))
        I_n = np.eye(n)
        J_r = ur3.jacobe(ur3.q)[3:, :]
        J_t_pinv = np.linalg.pinv(J_t)
        N = I_n - J_t_pinv @ J_t

        q_ee = sm.UnitQuaternion(R_ee).vec
        x_ee = R_ee[:, 0]
        z_ee = R_ee[:, 2]
        q_desired = np.array([1, 0, 0, 0])

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
            R_next = sm.UnitQuaternion(q_next).R
            x_next = R_next[:, 0]
            x_desired = np.array([0, 0, -1])
            penalty_x = lambda_penalty * np.linalg.norm(x_next - x_desired)**2
            return manip_term + penalty_quat + penalty_x
        

        def constraint_s2_ineq(q_dot_null):
            q_dot_total = q1 + N @ q_dot_null
            return b_ineq - A_ineq @ q_dot_total

        constraints_s2 = [{'type': 'ineq', 'fun': constraint_s2_ineq}]
        q_dot_null0 = np.zeros(n)
        result_s2 = minimize(objective_s2, q_dot_null0, constraints=constraints_s2)
        if not result_s2.success:
            print(f"S2 optimization failed: {result_s2.message}")

        qd = q1 + N @ result_s2.x

        # Orientation evaluation
        world_xz_normal = np.array([0, 1, 0])
        ee_xz_normal = np.cross(x_ee, z_ee)
        ee_xz_normal /= np.linalg.norm(ee_xz_normal)
        cos_theta_xz = np.dot(ee_xz_normal, world_xz_normal)
        cos_theta_xz = np.clip(cos_theta_xz, -1.0, 1.0)
        theta_xz_deg = np.arccos(abs(cos_theta_xz)) * 180 / np.pi

        # Debug output
        print(f"Angle between EE X-Z plane and world X-Z plane: {theta_xz_deg:.2f}°")

        # Apply joint velocities
        ur3.qd = qd
        env.step(dt)

        # Store data
        M = ur3.manipulability(ur3.q)
        print(f"Manipulability measure: {M}")
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
}
with open('manipulability_data_Boyu.json', 'w') as f:
    json.dump(data_to_save, f)

print("Data saved to manipulability_data_Boyu.json")