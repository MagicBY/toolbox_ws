#!/usr/bin/env python3
"""
Baseline test: DQ logarithmic-mapping motion tracking using rtb.p_servo
UR3 in Swift – straight vertical line with constant orientation
Tested with spatialmath-python ≥ 1.1.9 and roboticstoolbox-python ≥ 1.1.0
"""

import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import json

from spatialmath.base import r2q

# ------------------------------------------------------------------
# Dual Quaternion utilities (exactly the same as in the MATLAB script)
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

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
env = swift.Swift()
env.launch(realtime=True)

ur3 = rtb.models.UR3()

# Initial and final configurations (exactly the ones you asked for)
q_start = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
q_goal  = np.array([0.7854, -1.0472, 1.3963, -1.5708, 0.8727, 2.0944])

ur3.q = q_start.copy()
env.add(ur3)

H_goal = ur3.fkine(q_goal)        # fixed desired pose

dt = 0.05
t  = 0.0
arrived = False

# Logging
log_t   = []
log_pos = []
log_man = []

print("Starting DQ baseline test...")

while not arrived:
    H_curr = ur3.fkine(ur3.q)                 # current EE pose (SE3)

    # ---- 1. p_servo gives us the desired spatial twist (base frame) ----
    V_spatial, arrived = rtb.p_servo(H_curr, H_goal, gain=0.05)
    # V_spatial = [vx, vy, vz, wx, wy, wz]

    # ---- 2. Predict the pose we would reach in one dt step ----
    H_next = H_curr * sm.SE3.Exp(V_spatial * dt)

    # ---- 3. Relative displacement expressed in the current EE frame (DQ) ----
    F_curr = se3_to_dq(H_curr)
    F_next = se3_to_dq(H_next)
    x_des  = dq_mult(dq_conj(F_curr), F_next)   # this is exactly TF^{current}_{next}

    # ---- 4. Logarithmic map → body twist in DQ convention [ω; v] ----
    twist_dq = (2.0 / dt) * dq_log(x_des)        # [ωx, ωy, ωz, vx, vy, vz]

    # ---- 5. Convert to roboticstoolbox twist order [v; ω] ----
    twist_rt = np.hstack((twist_dq[3:], twist_dq[:3]))

    # ---- 6. Resolved-rate control ----
    J  = ur3.jacobe(ur3.q)
    qd = np.linalg.pinv(J) @ twist_rt
    ur3.qd = qd

    # ---- 7. Step simulator ----
    env.step(dt)
    t += dt

    # ---- Logging ----
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
}
with open("ur3_dq_baseline_data.json", "w") as f:
    json.dump(data, f)

print(f"Done in {t:.2f} s – reached goal!")
print("Data saved to ur3_dq_baseline_data.json")