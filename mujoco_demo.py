import time
import re
import shutil
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


def smooth_traj(t: float, T: float, q0: np.ndarray, qf: np.ndarray) -> np.ndarray:
    """Cosine interpolation (zero velocity at start/end)."""
    s = 0.5 * (1.0 - np.cos(np.pi * t / T))
    return q0 + s * (qf - q0)


def ensure_basename_meshes_exist(urdf_path: Path) -> None:
    """
    Workaround for some URDF import path behaviors:
    If URDF references meshes as 'meshes/foo.STL', MuJoCo may try to open 'foo.STL'
    in the URDF directory. This copies each referenced mesh into the URDF directory
    with its basename if missing.
    """
    txt = urdf_path.read_text(encoding="utf-8", errors="ignore")
    mesh_refs = re.findall(r'filename="([^"]+)"', txt)

    for ref in mesh_refs:
        src = (urdf_path.parent / ref).resolve()
        dst = urdf_path.parent / Path(ref).name  # basename in URDF folder
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)
            print(f"[mesh-fix] Copied {src.name} -> {dst.name}")


def main():
    # ------------------------------------------------------------
    # 0) Paths (URDF assumed next to this script)
    # ------------------------------------------------------------
    HERE = Path(__file__).resolve().parent
    URDF = HERE / "GEN3-7DOF-NOVISION_FOR_URDF_ARM_V12.urdf"  # <- your URDF name

    print("script dir :", HERE)
    print("cwd        :", Path.cwd())
    print("urdf exists:", URDF.exists(), URDF)

    if not URDF.exists():
        raise FileNotFoundError(f"URDF not found: {URDF}")

    # ------------------------------------------------------------
    # 1) (Optional) Fix mesh loading by ensuring basenames exist
    # ------------------------------------------------------------
    ensure_basename_meshes_exist(URDF)

    # ------------------------------------------------------------
    # 2) Load URDF
    # ------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(URDF))
    data = mujoco.MjData(model)

    print("Loaded model. nq =", model.nq, "nv =", model.nv, "nu =", model.nu)
    # For URDF imports, nu is often 0 (no actuators). That's OK for kinematic playback.

    # ------------------------------------------------------------
    # 3) Trajectory parameters
    # ------------------------------------------------------------
    arm_n = 7  # Kinova Gen3 arm joints
    dt = model.opt.timestep
    T = 3.0
    steps = int(T / dt)

    if model.nq < arm_n:
        raise RuntimeError(f"Model has nq={model.nq} (< {arm_n}). Unexpected for Gen3.")

    q_start = np.zeros(arm_n)
    q_goal = np.array([0.4, -0.6, 0.3, -1.0, 0.6, 1.0, 0.2], dtype=float)

    t_vec = np.linspace(0.0, T, steps)

    q_des = np.vstack([smooth_traj(t, T, q_start, q_goal) for t in t_vec])
    qd_des = np.zeros_like(q_des)
    qd_des[1:] = np.diff(q_des, axis=0) / dt

    # ------------------------------------------------------------
    # 4) Initialize state
    # ------------------------------------------------------------
    data.qpos[:arm_n] = q_start
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # 5) Run kinematic playback (Option A)
    # ------------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 1.5
        for i in range(steps):
            data.qpos[:arm_n] = q_des[i]
            data.qvel[:arm_n] = qd_des[i]  # optional; helps derived quantities
            mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()