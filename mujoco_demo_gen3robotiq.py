"""
Gen3 + Robotiq-85 in MuJoCo: control by EEF pose sequence + 3-phase gripper (explicit mimic)

What this does:
- Loops forever through an arbitrary EEF pose waypoint sequence (position + quaternion)
- Tracks EEF pose with damped-least-squares IK each frame
- Controls Robotiq-85 gripper with a 3-phase command (+1 open, -1 close, 0 hold)
- FIX: MuJoCo often ignores URDF <mimic>, so we explicitly drive the mimic joints so BOTH fingers move

Install:
  pip install mujoco numpy

Run:
  python gen3_mujoco_eef_ik_gripper_loop.py
"""
import os
import time
import numpy as np
import mujoco
import mujoco.viewer


# ======== UPDATE THIS PATH ON YOUR MACHINE ========
URDF_PATH = r"D:\mujoco_simulation\gen3_2f85.urdf"
os.chdir(os.path.dirname(URDF_PATH))


# -----------------------------
# Quaternion / rotation helpers
# -----------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # [w,x,y,z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return quat_normalize(q0 + s * (q1 - q0))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - s) * theta) / sin_theta
    b = np.sin(s * theta) / sin_theta
    return quat_normalize(a * q0 + b * q1)


def mat3_from_xmat(xmat9: np.ndarray) -> np.ndarray:
    return np.array(xmat9, dtype=float).reshape(3, 3)


def quat_from_mat3(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z], dtype=float))


def rotvec_from_quat(q_err: np.ndarray) -> np.ndarray:
    q_err = quat_normalize(q_err)
    w = float(np.clip(q_err[0], -1.0, 1.0))
    v = q_err[1:]
    sin_half = np.linalg.norm(v)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=float)
    axis = v / sin_half
    angle = 2.0 * np.arctan2(sin_half, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return axis * angle


# -----------------------------
# MuJoCo helpers
# -----------------------------
def _get_joint_ids_by_name(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    joint_ids: list[int] = []
    missing: list[str] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            missing.append(name)
        else:
            joint_ids.append(int(jid))
    if missing:
        all_joint_names = []
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm:
                all_joint_names.append(nm)
        raise RuntimeError(
            "Missing joints:\n"
            f"  {missing}\n\n"
            "Example available joints:\n"
            "  " + ", ".join(all_joint_names[:80]) + (" ..." if len(all_joint_names) > 80 else "")
        )
    return joint_ids


def find_ee_target(model: mujoco.MjModel) -> tuple[int, str]:
    """
    Prefer a body that corresponds to the gripper base / tool frame.
    Returns (body_id, name)
    """
    candidates = [
        "gen3_robotiq_85_base_link",
        "gen3_end_effector_link",
        "gen3_bracelet_link",
    ]
    for nm in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            return int(bid), nm
    last_body = model.nbody - 1
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, last_body) or f"body_{last_body}"
    return int(last_body), nm


def clamp_to_joint_limits(model: mujoco.MjModel, joint_ids: list[int], q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q[i] = float(np.clip(q[i], lo, hi))
    return q


# -----------------------------
# IK (damped least squares)
# -----------------------------
def get_body_pose(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    p = data.xpos[body_id].copy()
    R = mat3_from_xmat(data.xmat[body_id].copy())
    q = quat_from_mat3(R)
    return p, q


def jacobian_6d_body(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return np.vstack([jacp, jacr])


def ik_step_dls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    arm_joint_ids: list[int],
    qpos_adrs: list[int],
    dof_adrs: list[int],
    p_des: np.ndarray,
    q_des: np.ndarray,
    pos_w: float = 1.0,
    rot_w: float = 0.6,
    damping: float = 2e-2,
    step_scale: float = 0.9,
) -> float:
    p_cur, q_cur = get_body_pose(model, data, body_id)

    e_pos = (p_des - p_cur) * pos_w
    q_err = quat_mul(q_des, quat_conj(q_cur))
    e_rot = rotvec_from_quat(q_err) * rot_w
    e6 = np.hstack([e_pos, e_rot])
    err = float(np.linalg.norm(e6))

    J = jacobian_6d_body(model, data, body_id)
    J_arm = J[:, dof_adrs]  # 6 x 7

    A = (J_arm @ J_arm.T) + (damping ** 2) * np.eye(6)
    y = np.linalg.solve(A, e6)
    dq = J_arm.T @ y

    q = np.array([data.qpos[adr] for adr in qpos_adrs], dtype=float)
    q_new = q + step_scale * dq
    q_new = clamp_to_joint_limits(model, arm_joint_ids, q_new)

    for adr, val in zip(qpos_adrs, q_new):
        data.qpos[adr] = float(val)

    mujoco.mj_forward(model, data)
    return err


# -----------------------------
# Gripper explicit mimic control
# -----------------------------
def build_gripper_controls(model: mujoco.MjModel):
    """
    Returns:
      grip_controls: list of (qpos_adr, lo, hi, multiplier, joint_name)
      lo_m, hi_m: master joint range
    """
    GRIP_MASTER = "gen3_robotiq_85_left_knuckle_joint"
    GRIP_MIMICS = [
        ("gen3_robotiq_85_right_knuckle_joint",       -1.0),
        ("gen3_robotiq_85_left_inner_knuckle_joint",  +1.0),
        ("gen3_robotiq_85_right_inner_knuckle_joint", -1.0),
        ("gen3_robotiq_85_left_finger_tip_joint",     -1.0),
        ("gen3_robotiq_85_right_finger_tip_joint",    +1.0),
    ]

    def _info(jname: str):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            return None
        jid = int(jid)
        adr = int(model.jnt_qposadr[jid])
        limited = bool(model.jnt_limited[jid])
        if limited:
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
        else:
            lo, hi = None, None
        return adr, lo, hi

    master = _info(GRIP_MASTER)
    if master is None:
        print(f"[WARN] Missing gripper master joint '{GRIP_MASTER}'. Gripper disabled.")
        return [], 0.0, 0.8

    adr_m, lo_m, hi_m = master
    if lo_m is None or hi_m is None:
        lo_m, hi_m = 0.0, 0.8  # URDF says [0, 0.8]

    controls = [(adr_m, lo_m, hi_m, 1.0, GRIP_MASTER)]
    for jname, mult in GRIP_MIMICS:
        info = _info(jname)
        if info is None:
            print(f"[WARN] Missing gripper mimic joint '{jname}' (skipping)")
            continue
        adr, lo, hi = info
        controls.append((adr, lo, hi, float(mult), jname))

    print(f"[INFO] Gripper master '{GRIP_MASTER}' range=[{lo_m:.3f}, {hi_m:.3f}]")
    print("[INFO] Gripper controlled joints:")
    for adr, lo, hi, mult, name in controls:
        if lo is None:
            lim = "unlimited"
        else:
            lo2, hi2 = (lo, hi) if hi >= lo else (hi, lo)
            lim = f"[{lo2:.3f},{hi2:.3f}]"
        print(f"  {name:40s} mult={mult:+.1f} limits={lim}")

    return controls, lo_m, hi_m


def apply_gripper(model: mujoco.MjModel, data: mujoco.MjData, controls, lo_m, hi_m, g01: float):
    """
    g01 in [0,1]: 0=closed, 1=open (by default).
    If you want opposite, change q_master mapping below.
    """
    if not controls:
        return

    # Master joint in [lo_m, hi_m]
    q_master = lo_m + g01 * (hi_m - lo_m)

    for adr, lo, hi, mult, name in controls:
        q = mult * q_master
        if lo is not None and hi is not None:
            lo2, hi2 = (lo, hi) if hi >= lo else (hi, lo)
            q = float(np.clip(q, lo2, hi2))
        data.qpos[adr] = float(q)

    mujoco.mj_forward(model, data)


# -----------------------------
# Main
# -----------------------------
def main():
    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Arm joints
    arm_joint_names = [f"gen3_joint_{i}" for i in range(1, 8)]
    arm_joint_ids = _get_joint_ids_by_name(model, arm_joint_names)
    qpos_adrs = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
    dof_adrs = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]

    # EEF body
    ee_body_id, ee_name = find_ee_target(model)
    print(f"[INFO] Using end-effector body '{ee_name}' (id={ee_body_id})")

    # Gripper controls
    grip_controls, lo_m, hi_m = build_gripper_controls(model)

    # Timing / IK
    dt = 1.0 / 60.0
    IK_ITERS = 10

    # Waypoints (longer distance)
    p0, q0 = get_body_pose(model, data, ee_body_id)

    def offset_pose(dp: np.ndarray, yaw_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        yaw = np.deg2rad(yaw_deg)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        q_yaw = np.array([cy, 0.0, 0.0, sy], dtype=float)
        return p0 + dp, quat_mul(q_yaw, q0)

    DX, DY, DZ = 0.26, 0.20, 0.14  # meters

    # (p, q, duration, gripper_cmd)
    waypoints = [
        (*offset_pose(np.array([0.00, 0.00, 0.00]), 0.0),   1.2,  0),
        (*offset_pose(np.array([DX,   0.00, 0.00]), 20.0),  2.0, +1),
        (*offset_pose(np.array([DX,   DY,   0.00]), 0.0),   2.0,  0),
        (*offset_pose(np.array([0.00, DY,   0.00]), -20.0), 2.0, -1),
        (*offset_pose(np.array([0.00, 0.00, DZ  ]), 0.0),   2.0,  0),
        (*offset_pose(np.array([DX*0.6, DY*0.3, DZ]), 10.0),2.0, +1),
        (*offset_pose(np.array([0.00, 0.00, 0.00]), 0.0),   2.0,  0),
    ]

    # Gripper open fraction g in [0,1]
    g = 0.8
    GRIP_SPEED = 0.9  # g units per second

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 1.8
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        t0 = time.perf_counter()
        k_global = 0

        try:
            while viewer.is_running():
                # iterate segments (including wrap-around)
                for seg in range(len(waypoints)):
                    pA, qA, TA, cmdA = waypoints[seg]
                    pB, qB, _, _ = waypoints[(seg + 1) % len(waypoints)]

                    T = float(TA)
                    n = max(2, int(np.ceil(T / dt)))

                    for i in range(n):
                        if not viewer.is_running():
                            break

                        s = i / (n - 1)
                        p_des = (1.0 - s) * pA + s * pB
                        q_des = quat_slerp(qA, qB, s)

                        # Gripper update (3-phase)
                        g += float(cmdA) * GRIP_SPEED * dt
                        g = float(np.clip(g, 0.0, 1.0))
                        apply_gripper(model, data, grip_controls, lo_m, hi_m, g)

                        # IK
                        for _ in range(IK_ITERS):
                            err = ik_step_dls(
                                model, data, ee_body_id,
                                arm_joint_ids, qpos_adrs, dof_adrs,
                                p_des, q_des,
                            )
                            if err < 1e-4:
                                break

                        viewer.sync()

                        # strict pacing
                        target = t0 + k_global * dt
                        while time.perf_counter() < target:
                            time.sleep(0.0005)
                        k_global += 1

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
