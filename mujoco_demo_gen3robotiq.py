"""
Kinova Gen3 URDF trajectory playback in MuJoCo (Python) — corrected version (looping)

Change requested:
- Keep moving continuously until you stop the script (Ctrl+C / close window).
- Loops the same 3-segment trajectory forever.
- Viewer stays open; handles Ctrl+C gracefully.

Install:
  pip install mujoco numpy

Run:
  python gen3_urdf_trajectory_demo.py
"""
import os
import time
import numpy as np
import mujoco
import mujoco.viewer


# ======== UPDATE THIS PATH ON YOUR MACHINE ========
URDF_PATH = r"D:\mujoco_simulation\gen3_2f85.urdf"
os.chdir(os.path.dirname(URDF_PATH))


def quintic_time_scaling(s: np.ndarray) -> np.ndarray:
    """Quintic polynomial (0->1) with zero vel/acc endpoints: 10s^3 - 15s^4 + 6s^5."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def make_segment(q0: np.ndarray, q1: np.ndarray, duration: float, dt: float) -> np.ndarray:
    """Generate a smooth joint trajectory from q0 to q1 over 'duration' seconds."""
    n = max(2, int(np.ceil(duration / dt)) + 1)
    t = np.linspace(0.0, duration, n)
    s = t / duration
    a = quintic_time_scaling(s)[:, None]  # (n,1)
    return (1 - a) * q0[None, :] + a * q1[None, :]


def _get_joint_ids_by_name(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    """Return MuJoCo joint ids for given joint names, error if any missing."""
    joint_ids = []
    missing = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            missing.append(name)
        joint_ids.append(jid)

    if missing:
        all_joint_names = []
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm is not None:
                all_joint_names.append(nm)

        raise RuntimeError(
            "Could not find these joints in the loaded model:\n"
            f"  {missing}\n\n"
            "Available joint names in this model include:\n"
            "  " + ", ".join(all_joint_names[:50]) + (" ..." if len(all_joint_names) > 50 else "")
        )

    return joint_ids


def main():
    # ---- Load model ----
    try:
        model = mujoco.MjModel.from_xml_path(URDF_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load URDF at:\n  {URDF_PATH}\n\n"
            f"MuJoCo error: {e}\n\n"
            "If this is a mesh/path issue, you can temporarily remove <visual>/<collision> "
            "or fix <mesh filename=...> paths to valid absolute/relative paths."
        )

    data = mujoco.MjData(model)

    # ---- Select Gen3 arm joints by name ----
    arm_joint_names = [f"gen3_joint_{i}" for i in range(1, 8)]
    arm_joint_ids = _get_joint_ids_by_name(model, arm_joint_names)

    # qpos indices (addresses) for these joints
    qpos_adrs = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]

    # ---- Trajectory ----
    dt = 1.0 / 60.0  # 60 Hz playback

    # Start from current default pose (often zeros)
    q_start = np.array([data.qpos[adr] for adr in qpos_adrs], dtype=float)

    # Two target poses (radians). Adjust freely.
    q_mid = q_start + np.array([0.25, -0.35, 0.20, -0.40, 0.25, 0.30, -0.20], dtype=float)
    q_end = q_start + np.array([-0.15, 0.20, -0.30, 0.15, -0.20, 0.10, 0.25], dtype=float)

    seg1 = make_segment(q_start, q_mid, duration=1.5, dt=dt)
    seg2 = make_segment(q_mid, q_end, duration=1.5, dt=dt)
    seg3 = make_segment(q_end, q_start, duration=1.5, dt=dt)
    q_traj = np.vstack([seg1, seg2[1:], seg3[1:]])  # remove duplicate boundary points

    # ---- Logging (keeps growing while running) ----
    t_list: list[float] = []
    q_list: list[np.ndarray] = []

    # ---- Viewer + playback (loop forever) ----
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 1.8
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        t0 = time.time()
        k_global = 0

        try:
            while viewer.is_running():
                for qk in q_traj:
                    if not viewer.is_running():
                        break

                    # Set qpos for arm joints
                    for adr, val in zip(qpos_adrs, qk):
                        data.qpos[adr] = float(val)

                    mujoco.mj_forward(model, data)

                    # log (time since start, and joint angles)
                    t_now = k_global * dt
                    t_list.append(t_now)
                    q_list.append(qk.copy())

                    viewer.sync()

                    # real-time pacing
                    target = t0 + t_now
                    while time.time() < target:
                        time.sleep(0.0005)

                    k_global += 1

        except KeyboardInterrupt:
            pass  # allow Ctrl+C to exit cleanly

    print(f"Stopped. Logged {len(q_list)} samples of 7-DoF joint angles.")
    if q_list:
        print("Last sample:", q_list[-1])


if __name__ == "__main__":
    main()
