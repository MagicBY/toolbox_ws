#!/usr/bin/env python3
"""
UR3 simulation in Swift: DQ-based desired trajectory with hierarchical manipulability optimization.
Uses rtb.p_servo for baseline twist, synthesizes via DQ, then optimizes in joint space with translation primary and manipulability + orientation secondary.
Tracks full pose (position + orientation) from H_goal.
"""

import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import json
from scipy.optimize import minimize
from spatialmath.base import r2q

# ------------------------------------------------------------------
# Dual Quaternion and Quaternion utilities
# ------------------------------------------------------------------
def quat_mult(q1, q2):
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    w = w1 * w2 - np.dot(v1, v2)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return np.hstack((w, v))

def dq_mult(dq1, dq2):
    p1, d1 = dq1[:4], dq1[4:]
    p2, d2 = dq2[:4], dq2[4:]
    return np.hstack((quat_mult(p1, p2),
                      quat_mult(p1, d2) + quat_mult(d1, p2)))

def dq_conj(dq):
    return np.hstack((dq[0], -dq[1:4], dq[4], -dq[5:8]))

def dq_exp(vec):
    r, t = vec[:3], vec[3:]
    nr = np.linalg.norm(r)
    if nr < 1e-9:
        gp = np.array([1, 0, 0, 0])
    else:
        gp = np.hstack((np.cos(nr/2), np.sin(nr/2) * (r / nr)))
    dual = 0.5 * quat_mult(np.hstack((0, t)), gp)
    return np.hstack((gp, dual))

def dq_log(dq):
    p = dq[:4] / np.linalg.norm(dq[:4])      # unit quaternion part
    d = dq[4:]
    w, v = p[0], p[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        rot = np.zeros(3)
    else:
        phi = 2 * np.arctan2(nv, w)
        rot = (phi / 2) * (v / nv)
    # translation part
    t = 2 * quat_mult(d, np.hstack((p[0], -p[1:])))
    screw = np.hstack((rot, t[1:]/2))  # initial log output

    # Check for shorter path (handle double-cover)
    x_check = dq_exp(screw)  # exp back to DQ
    id_pos = np.array([1, 0, 0, 0])
    id_neg = np.array([-1, 0, 0, 0])
    if np.linalg.norm(x_check[:4] - id_pos) > np.linalg.norm(x_check[:4] - id_neg):
        screw = dq_log(-x_check)  # flip sign and re-log for shorter twist

    return screw

def se3_to_dq(T: sm.SE3):
    """Convert spatialmath SE3 → unit dual quaternion [real; dual]"""
    quat = r2q(T.R)                      # [w, x, y, z] scalar-first
    t    = T.t
    dual = 0.5 * quat_mult(np.hstack((0, t)), quat)
    return np.hstack((quat, dual))

def quat_conj(q):
    return np.hstack((q[0], -q[1:]))

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
env = swift.Swift()
env.launch(realtime=True)

ur3 = rtb.models.UR3()

# Initial and final configurations
q_start = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
q_goal  = np.array([0.7854, -1.0472, 1.3963, -1.5708, 0.8727, 2.0944])

ur3.q = q_start.copy()
env.add(ur3)

H_goal = ur3.fkine(q_goal)        # fixed desired pose
q_des = r2q(H_goal.R)             # desired quaternion from H_goal

# Parameters
n = 6
dt = 0.05
gain = 0.05  # p_servo gain
lambda_penalty = 10  # orientation penalty weight
t = 0.0
arrived = False

# Joint velocity limits
q_dot_max = np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
q_dot_min = -q_dot_max
A_ineq = np.vstack((np.eye(n), -np.eye(n)))
b_ineq = np.hstack((q_dot_max, -q_dot_min))

# Logging
log_t = []
log_pos = []
log_man = []
log_vel_err = []  # ||J_t @ qd - v||
log_theta_e = []  # orientation error angle

print("Starting DQ-optimized simulation...")

while not arrived:
    H_curr = ur3.fkine(ur3.q)  # current EE pose (SE3)
    R_ee = H_curr.R
    q_ee = r2q(R_ee)  # current quaternion

    # ---- DQ-based desired trajectory command ----
    V_spatial, arrived = rtb.p_servo(H_curr, H_goal, gain=gain)
    H_next = H_curr * sm.SE3.Exp(V_spatial * dt)
    F_curr = se3_to_dq(H_curr)
    F_next = se3_to_dq(H_next)
    x_des = dq_mult(dq_conj(F_curr), F_next)
    twist_dq = (2.0 / dt) * dq_log(x_des)  # [ω; v]
    twist_rt = np.hstack((twist_dq[3:], twist_dq[:3]))  # [v; ω] for rtb

    v_des = twist_rt[:3]  # desired linear velocity
    omega_des = twist_rt[3:]  # desired angular velocity (from DQ)

    # Jacobians
    J = ur3.jacobe(ur3.q)
    J_t = J[:3, :]  # translational
    J_r = J[3:, :]  # rotational

    # ---- S1: Translational task optimization ----
    def objective_s1(q_dot):
        return 0.5 * np.linalg.norm(J_t @ q_dot - v_des)**2

    result_s1 = minimize(objective_s1, ur3.qd)
    q1 = result_s1.x

    # Null space projector
    J_t_pinv = np.linalg.pinv(J_t)
    N = np.eye(n) - J_t_pinv @ J_t

    # Manipulability gradient
    Jm = ur3.jacobm(end=ur3.ee_links[0]).reshape((n,))

    # ---- S2: Null space optimization (manipulability + orientation) ----
    def objective_s2(q_dot_null):
        q_dot_total = q1 + N @ q_dot_null
        manip_term = 0.5 * np.linalg.norm(q_dot_total - Jm)**2  # maximize manip (note: positive for max via gradient ascent emulation)
        # Orientation penalty (predict next quat)
        omega = J_r @ q_dot_total
        q_omega = np.hstack((0, omega))
        dq_quat = 0.5 * quat_mult(q_ee, q_omega)
        q_next = q_ee + dt * dq_quat
        q_next /= np.linalg.norm(q_next)
        q_error_next = quat_mult(quat_conj(q_des), q_next)
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
    ur3.qd = qd

    # Evaluations
    vel_err = np.linalg.norm(J_t @ qd - v_des)
    log_vel_err.append(vel_err)

    q_error = quat_mult(quat_conj(q_des), q_ee)
    w_e = q_error[0]
    theta_e = 2 * np.arccos(np.clip(w_e, -1.0, 1.0))
    log_theta_e.append(theta_e)

    # Step simulator
    env.step(dt)
    t += dt

    # Logging
    log_t.append(t)
    log_pos.append(H_curr.t.tolist())
    log_man.append(ur3.manipulability(ur3.q))

# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------
data = {
    "time": log_t,
    "position": log_pos,
    "manipulability": log_man,
    "velocity_error": log_vel_err,
    "theta_e": log_theta_e,
}
with open("ur3_dq_optimized_data.json", "w") as f:
    json.dump(data, f)

print(f"Done in {t:.2f} s – reached goal!")
print("Data saved to ur3_dq_optimized_data.json")