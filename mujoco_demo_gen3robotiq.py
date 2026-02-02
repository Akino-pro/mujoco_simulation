"""
Kinova Gen3 URDF trajectory playback in MuJoCo (Python)

- Loads a URDF (your gen3_2f85_urdf)
- Detects the first 7 hinge joints (arm joints)
- Generates a short smooth joint-space trajectory (quintic time scaling between waypoints)
- Plays it back in a viewer
- Stores joint angles over time in a list

Install:
  pip install mujoco numpy

Run:
  python gen3_urdf_trajectory_demo.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer


# ======== UPDATE THIS PATH ON YOUR MACHINE ========
URDF_PATH = r"D:\github\gen3_2f85.urdf"
# On Windows you might use:
# URDF_PATH = r"D:\path\to\gen3_2f85.urdf"


def quintic_time_scaling(s: np.ndarray) -> np.ndarray:
    """Quintic polynomial (0->1) with zero vel/acc endpoints: 10s^3 - 15s^4 + 6s^5."""
    return 10*s**3 - 15*s**4 + 6*s**5


def make_segment(q0: np.ndarray, q1: np.ndarray, duration: float, dt: float) -> np.ndarray:
    """Generate a smooth joint trajectory from q0 to q1 over 'duration' seconds."""
    n = max(2, int(np.ceil(duration / dt)) + 1)
    t = np.linspace(0.0, duration, n)
    s = t / duration
    a = quintic_time_scaling(s)[:, None]  # (n,1)
    return (1 - a) * q0[None, :] + a * q1[None, :]


def main():
    # ---- Load model ----
    try:
        model = mujoco.MjModel.from_xml_path(URDF_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load URDF at:\n  {URDF_PATH}\n\n"
            f"MuJoCo error: {e}\n\n"
            "If this is a mesh/path issue, you can temporarily comment out visuals in the URDF "
            "or fix <mesh filename=...> paths to absolute/relative paths that exist."
        )

    data = mujoco.MjData(model)

    # ---- Identify the first 7 hinge joints (typical Gen3 arm joints) ----
    hinge_joints = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_joints.append(j)

    if len(hinge_joints) < 7:
        raise RuntimeError(
            f"Found only {len(hinge_joints)} hinge joints, expected at least 7 for Gen3.\n"
            "Your URDF may be different (or joints imported differently)."
        )

    arm_joints = hinge_joints[:7]

    # qpos indices for these joints
    qpos_adrs = [int(model.jnt_qposadr[j]) for j in arm_joints]

    # ---- Build a short demo trajectory (3 segments) ----
    dt = 1.0 / 60.0  # 60 Hz playback

    # Start from current default pose (often zeros)
    q_start = np.array([data.qpos[adr] for adr in qpos_adrs], dtype=float)

    # Two target poses (radians). Adjust freely.
    q_mid = q_start + np.array([0.25, -0.35, 0.20, -0.40, 0.25, 0.30, -0.20])
    q_end = q_start + np.array([-0.15, 0.20, -0.30, 0.15, -0.20, 0.10, 0.25])

    # Create segments with smooth time scaling
    seg1 = make_segment(q_start, q_mid, duration=1.5, dt=dt)
    seg2 = make_segment(q_mid, q_end, duration=1.5, dt=dt)
    seg3 = make_segment(q_end, q_start, duration=1.5, dt=dt)

    q_traj = np.vstack([seg1, seg2[1:], seg3[1:]])  # remove duplicate boundary points

    # Optional: store full (t, q) history
    t_list = []
    q_list = []

    # ---- Viewer + playback ----
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 1.8  # zoom out a bit (bigger -> farther)
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        t0 = time.time()
        for k in range(len(q_traj)):
            qk = q_traj[k]

            # Set qpos for arm joints
            for adr, val in zip(qpos_adrs, qk):
                data.qpos[adr] = val

            mujoco.mj_forward(model, data)

            # log
            t_now = k * dt
            t_list.append(t_now)
            q_list.append(qk.copy())

            viewer.sync()

            # real-time pacing
            target = t0 + t_now
            while time.time() < target:
                time.sleep(0.0005)

    # Now you have your list of joint angles over time:
    # q_list: List[np.ndarray] length T, each shape (7,)
    print(f"Trajectory finished. Stored {len(q_list)} samples of 7-DoF joint angles.")
    print("Example first sample:", q_list[0])
    print("Example last sample: ", q_list[-1])


if __name__ == "__main__":
    main()
