import os
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import mujoco
import mujoco.viewer

from helper_functions import (
    _get_joint_ids_by_name, find_ee_target, build_gripper_controls, quat_from_mat3,
    set_white_environment_visuals, apply_gripper, show_wrist_window,
    depth_to_vis
)

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    import cv2
except Exception:
    cv2 = None


URDF_PATH = r"gen3_modified.urdf"

# Your trained checkpoint
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = SCRIPT_DIR / "ACTmodel"/"pretrained_model"

REPO_ID = "lsnu/TeddyBearKinovaTestSetLeRobot"

# Bad sample indices discovered from full scan
BAD_IDX = [
    7068, 7069, 7070, 7071, 7072, 7073, 7074,
    14448, 14449, 14450, 14451, 14452, 14453, 14454, 14455, 14456,
    21707, 21708, 21709, 21710, 21711, 21712,
    25521, 25522
]

# Task text fed to policy
TASK_TEXT = "pick and place the large teddy bear into the cardboard box"


def _resolve_scene_path(urdf_dir: str) -> str:
    env_scene = os.environ.get("MUJOCO_SCENE_XML", "").strip()
    if env_scene:
        scene = env_scene if os.path.isabs(env_scene) else os.path.join(urdf_dir, env_scene)
        if not os.path.exists(scene):
            raise FileNotFoundError(f"MUJOCO_SCENE_XML points to missing file: {scene}")
        print(f"[INFO] Using scene from MUJOCO_SCENE_XML: {scene}")
        return scene

    convex_scene = os.path.join(urdf_dir, "robotsuit_convex.xml")
    cubes_scene = os.path.join(urdf_dir, "robotsuit_cubes.xml")
    if os.path.exists(convex_scene):
        print(f"[INFO] Using convex scene: {convex_scene}")
        return convex_scene
    print(f"[INFO] Using cube-proxy scene: {cubes_scene}")
    return cubes_scene


def _resolve_collision_viz_choice() -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--collision-viz",
        type=str,
        default="",
        help="Collision visualization mode: 1/off/hide or 2/on/show.",
    )
    args, _ = parser.parse_known_args()
    raw = str(args.collision_viz).strip().lower()

    if raw in {"1", "off", "hide"}:
        return False
    if raw in {"2", "on", "show"}:
        return True

    print("[SELECT] Collision visualization mode:")
    print("  1) Hide collision boxes/meshes")
    print("  2) Show collision boxes/meshes (current behavior)")
    while True:
        choice = input("Enter 1 or 2 (default 2): ").strip()
        if choice == "":
            return True
        if choice == "1":
            return False
        if choice == "2":
            return True
        print("Please enter 1 or 2.")


def _apply_collision_viz_mode(model: mujoco.MjModel, show_collision_viz: bool):
    if show_collision_viz:
        return
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not gname:
            continue
        if "_col" in gname:
            model.geom_rgba[gid, 3] = 0.0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def mat3_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_to_rotvec(q_err: np.ndarray) -> np.ndarray:
    v = np.zeros(3, dtype=float)
    mujoco.mju_quat2Vel(v, q_err, 1.0)
    return v


def dls_ik_compute_qtarget(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_body_id: int,
    arm_joint_ids: list[int],
    qpos_adrs: list[int],
    dof_adrs: list[int],
    p_des: np.ndarray,
    q_des: np.ndarray,
    rot_w: float,
    damping: float,
    dt_step: float,
    qd_max: np.ndarray,
    iters: int,
) -> np.ndarray:
    q_arm0 = np.array([data.qpos[a] for a in qpos_adrs], dtype=float)

    for _ in range(iters):
        mujoco.mj_forward(model, data)

        p_cur = data.xpos[ee_body_id].copy()
        R_cur = data.xmat[ee_body_id].reshape(3, 3).copy()
        q_cur = mat3_to_quat(R_cur)

        ep = (p_des - p_cur)
        q_err = quat_mul(q_des, quat_conj(q_cur))
        er = quat_to_rotvec(q_err)
        e = np.concatenate([ep, rot_w * er], axis=0)

        Jp = np.zeros((3, model.nv), dtype=float)
        Jr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacBody(model, data, Jp, Jr, ee_body_id)

        Jp7 = Jp[:, dof_adrs]
        Jr7 = Jr[:, dof_adrs]
        J6 = np.vstack([Jp7, rot_w * Jr7])

        JJt = J6 @ J6.T + (damping ** 2) * np.eye(6)
        try:
            x = np.linalg.solve(JJt, e)
        except np.linalg.LinAlgError:
            break
        dq = J6.T @ x

        dq_lim = qd_max * dt_step
        dq = np.clip(dq, -dq_lim, dq_lim)

        for i, adr in enumerate(qpos_adrs):
            data.qpos[adr] += dq[i]

        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 2e-4:
            break

    q_target = np.array([data.qpos[a] for a in qpos_adrs], dtype=float)

    for i, adr in enumerate(qpos_adrs):
        data.qpos[adr] = q_arm0[i]
    mujoco.mj_forward(model, data)

    return q_target


def clamp_vec(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def limit_step(target: np.ndarray, current: np.ndarray, max_delta: np.ndarray) -> np.ndarray:
    delta = np.clip(target - current, -max_delta, max_delta)
    return current + delta


def euler_xyz_deg_to_mat3(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ], dtype=float)

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=float)

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [0,   0,  1],
    ], dtype=float)

    # XYZ-euler convention implemented as Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def mat3_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
    # Inverse of R = Rz @ Ry @ Rx
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    ry = np.arcsin(sy)

    cy = np.cos(ry)
    if abs(cy) > 1e-8:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal-lock fallback
        rx = 0.0
        rz = np.arctan2(-R[0, 1], R[1, 1])

    return np.rad2deg(np.array([rx, ry, rz], dtype=float))


def render_camera_rgb(renderer, data, camera_name: str) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()
    return img.copy()


def build_policy_raw_obs(wrist_rgb: np.ndarray, fixed_rgb: np.ndarray, eef_state: np.ndarray, task_text: str):
    # LeRobot dataset samples were CHW torch tensors, so convert HWC->CHW
    wrist_t = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float()
    fixed_t = torch.from_numpy(fixed_rgb).permute(2, 0, 1).float()
    state_t = torch.from_numpy(eef_state).float()

    return {
        "observation.images.wrist": wrist_t.unsqueeze(0),
        "observation.images.azure_rgb": fixed_t.unsqueeze(0),
        "observation.state": state_t.unsqueeze(0),
        "task": [task_text],
    }


def build_clean_dataset_and_processors(policy: ACTPolicy):
    base_ds = LeRobotDataset(REPO_ID, tolerance_s=0.02)
    bad_episodes = sorted({int(base_ds.hf_dataset[i]["episode_index"]) for i in BAD_IDX})
    good_episodes = [ep for ep in range(base_ds.num_episodes) if ep not in bad_episodes]

    clean_ds = LeRobotDataset(
        REPO_ID,
        episodes=good_episodes,
        tolerance_s=0.02,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=clean_ds.meta.stats,
    )
    return clean_ds, preprocessor, postprocessor


def main():
    show_collision_viz = _resolve_collision_viz_choice()
    print(f"[INFO] Collision visualization: {'ON' if show_collision_viz else 'OFF'}")

    urdf_abs = os.path.abspath(URDF_PATH)
    urdf_dir = os.path.dirname(urdf_abs)
    os.chdir(urdf_dir)

    scene_path = _resolve_scene_path(urdf_dir)
    model = mujoco.MjModel.from_xml_path(scene_path)
    _apply_collision_viz_mode(model, show_collision_viz)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    wrist_cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgb"))
    fixed_cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_down"))
    if wrist_cam_id < 0 or fixed_cam_id < 0:
        raise RuntimeError("Missing wrist_rgb or fixed_down camera in XML.")

    arm_joint_names = [f"gen3_joint_{i}" for i in range(1, 8)]
    arm_joint_ids = _get_joint_ids_by_name(model, arm_joint_names)
    qpos_adrs = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
    dof_adrs = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]

    # Hold immediately
    data.ctrl[:7] = np.array([data.qpos[a] for a in qpos_adrs], dtype=float)

    ee_body_id, ee_name = find_ee_target(model)
    print(f"[INFO] Using end-effector body '{ee_name}' (id={ee_body_id})")

    grip_controls, lo_m, hi_m = build_gripper_controls(model)

    bracelet_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gen3_bracelet_link")
    if bracelet_body_id < 0:
        raise RuntimeError("Could not find body 'gen3_bracelet_link'.")
    bracelet_body_id = int(bracelet_body_id)

    cam_off = np.array([0.0, -0.06841, -0.05044], dtype=float)

    renderer = None
    if cv2 is not None:
        renderer = mujoco.Renderer(model, width=960, height=540)

    renderer_depth = None
    if cv2 is not None:
        renderer_depth = mujoco.Renderer(model, width=960, height=540)
        renderer_depth.enable_depth_rendering()

    dt_view = 1.0 / 60.0
    n_sub_transfer = max(1, int(0.5 * round(dt_view / model.opt.timestep)))

    IK_ITERS_TRANSFER = 6
    qd_safe = np.deg2rad(30.0)
    qd_max = np.full(7, qd_safe, dtype=float)

    # Friction tweaks from your original script
    def _set_geom_friction(name: str, slide=3.0, spin=0.03, roll=0.002):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_friction[int(gid), :] = np.array([slide, spin, roll], dtype=float)

    def _set_geom_friction_prefix(prefix: str, slide=3.0, spin=0.03, roll=0.002):
        for gid in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and gname.startswith(prefix):
                model.geom_friction[gid, :] = np.array([slide, spin, roll], dtype=float)

    _set_geom_friction("big_bear_col", slide=4.0, spin=0.04, roll=0.003)
    _set_geom_friction_prefix("big_bear_col_part_", slide=4.0, spin=0.04, roll=0.003)
    _set_geom_friction("cardboard_col", slide=3.2, spin=0.03, roll=0.002)
    _set_geom_friction_prefix("cardboard_col_part_", slide=3.2, spin=0.03, roll=0.002)
    _set_geom_friction("left_fingertip_col", slide=4.0, spin=0.04, roll=0.003)
    _set_geom_friction("right_fingertip_col", slide=4.0, spin=0.04, roll=0.003)

    # Policy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(CKPT_PATH)
    policy.eval()
    policy.to(device)
    print(f"[INFO] Loaded ACT checkpoint from: {CKPT_PATH}")
    print(f"[INFO] Policy device: {device}")

    clean_ds, preprocessor, postprocessor = build_clean_dataset_and_processors(policy)
    print(f"[INFO] Clean dataset episodes: {clean_ds.num_episodes}, frames: {clean_ds.num_frames}")

    # Your dataset gripper distribution
    gripper_min = 0.8733651041984558
    gripper_max = 100.0

    # Safety bounds for EEF command
    xyz_lo = np.array([-0.80, -0.80, 0.60], dtype=float)
    xyz_hi = np.array([ 0.80,  0.80, 1.20], dtype=float)
    rpy_lo = np.array([-180.0, -180.0, -180.0], dtype=float)
    rpy_hi = np.array([ 180.0,  180.0,  180.0], dtype=float)

    # Step limits for smoothness
    max_xyz_delta = np.array([0.01, 0.01, 0.01], dtype=float)   # meters per control step
    max_rpy_delta = np.array([2.0, 2.0, 2.0], dtype=float)      # deg per control step

    # If you want slower policy updates, increase this
    policy_every = 1

    # Render depth settings
    MAX_DEPTH_WRIST = 1.0
    MAX_DEPTH_FIXED = 2.5

    next_step_time = time.perf_counter()

    def step_realtime():
        nonlocal next_step_time
        next_step_time += dt_view
        now = time.perf_counter()
        if now - next_step_time > 0.2:
            next_step_time = now
        sleep_s = next_step_time - now
        if sleep_s > 0:
            time.sleep(sleep_s)

    def update_wrist_camera_pose():
        p_b = data.xpos[bracelet_body_id].copy()
        R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
        p_frame = p_b + R_b @ cam_off

        R_frame = data.xmat[ee_body_id].reshape(3, 3).copy()
        tool_forward = np.array([0.0, 0.0, -1.0], dtype=float)
        tool_up = np.array([0.0, -1.0, 0.0], dtype=float)

        eye_out = 0.01
        p_frame = p_frame + (R_frame @ (tool_forward * eye_out))

        f_world = _normalize(R_frame @ tool_forward)
        u_world = _normalize(R_frame @ tool_up)

        z_cam_world = _normalize(-f_world)
        y_cam_world = _normalize(u_world - np.dot(u_world, z_cam_world) * z_cam_world)
        x_cam_world = _normalize(np.cross(y_cam_world, z_cam_world))
        y_cam_world = _normalize(np.cross(z_cam_world, x_cam_world))

        R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])

        model.cam_pos[wrist_cam_id] = p_frame
        model.cam_quat[wrist_cam_id] = quat_from_mat3(R_wc)

    def render_windows():
        if renderer is None:
            return

        renderer.update_scene(data, camera="wrist_rgb")
        wrist_rgb_vis = renderer.render()
        show_wrist_window(wrist_rgb_vis[..., ::-1], title="Wrist Camera (RGB)", w=960, h=540)

        renderer.update_scene(data, camera="fixed_down")
        fixed_rgb_vis = renderer.render()
        show_wrist_window(fixed_rgb_vis[..., ::-1], title="Fixed Camera (RGB)", w=960, h=540)

        if renderer_depth is not None:
            renderer_depth.update_scene(data, camera="wrist_rgb")
            wrist_depth = renderer_depth.render()
            wrist_depth_vis = depth_to_vis(wrist_depth, max_depth=MAX_DEPTH_WRIST, use_colormap=True)
            show_wrist_window(wrist_depth_vis, title="Wrist Camera (Depth)", w=960, h=540)

            renderer_depth.update_scene(data, camera="fixed_down")
            fixed_depth = renderer_depth.render()
            fixed_depth_vis = depth_to_vis(fixed_depth, max_depth=MAX_DEPTH_FIXED, use_colormap=True)
            show_wrist_window(fixed_depth_vis, title="Fixed Camera (Depth)", w=960, h=540)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 1.8
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        set_white_environment_visuals(model, viewer)

        # Initialize from current pose
        mujoco.mj_forward(model, data)
        p_cur = data.xpos[ee_body_id].copy()
        R_cur = data.xmat[ee_body_id].reshape(3, 3).copy()
        pred_rpy_deg = mat3_to_euler_xyz_deg(R_cur)

        # Start gripper half-open in raw units mapped to sim [0,1]
        g_raw = gripper_min
        g = (g_raw - gripper_min) / (gripper_max - gripper_min)
        g = float(np.clip(g, 0.0, 1.0))

        step_count = 0

        try:
            while viewer.is_running():
                mujoco.mj_forward(model, data)

                p_cur = data.xpos[ee_body_id].copy()
                R_cur = data.xmat[ee_body_id].reshape(3, 3).copy()
                cur_rpy_deg = mat3_to_euler_xyz_deg(R_cur)

                # Build observation state in dataset action/state format
                eef_state = np.array([
                    p_cur[0], p_cur[1], p_cur[2],
                    cur_rpy_deg[0], cur_rpy_deg[1], cur_rpy_deg[2],
                    g * (gripper_max - gripper_min) + gripper_min,
                ], dtype=np.float32)

                if (step_count % policy_every) == 0:
                    update_wrist_camera_pose()

                    wrist_rgb = render_camera_rgb(renderer, data, "wrist_rgb")
                    fixed_rgb = render_camera_rgb(renderer, data, "fixed_down")

                    raw_obs = build_policy_raw_obs(
                        wrist_rgb=wrist_rgb,
                        fixed_rgb=fixed_rgb,
                        eef_state=eef_state,
                        task_text=TASK_TEXT,
                    )

                    with torch.no_grad():
                        processed_obs = preprocessor(raw_obs)
                        for k, v in processed_obs.items():
                            if isinstance(v, torch.Tensor):
                                processed_obs[k] = v.to(device)

                        pred_action_norm = policy.select_action(processed_obs)
                        pred_action_denorm = postprocessor(pred_action_norm)

                    pred = pred_action_denorm.squeeze(0).detach().cpu().numpy()

                    # Predicted raw action
                    pred_xyz = pred[:3].astype(float)
                    pred_rpy_raw = pred[3:6].astype(float)
                    pred_gripper_raw = float(pred[6])

                    # Safety clamp
                    pred_xyz = clamp_vec(pred_xyz, xyz_lo, xyz_hi)
                    pred_rpy_raw = clamp_vec(pred_rpy_raw, rpy_lo, rpy_hi)
                    pred_gripper_raw = float(np.clip(pred_gripper_raw, gripper_min, gripper_max))

                    # Smooth per-step limiting
                    pred_xyz = limit_step(pred_xyz, p_cur, max_xyz_delta)
                    pred_rpy_deg = limit_step(pred_rpy_raw, cur_rpy_deg, max_rpy_delta)

                    R_des = euler_xyz_deg_to_mat3(
                        pred_rpy_deg[0], pred_rpy_deg[1], pred_rpy_deg[2]
                    )
                    q_des = quat_from_mat3(R_des)

                    # Map raw dataset gripper -> sim [0,1]
                    g = (pred_gripper_raw - gripper_min) / (gripper_max - gripper_min)
                    g = float(np.clip(g, 0.0, 1.0))

                    if (step_count % 30) == 0:
                        print(
                            f"[POLICY] xyz={pred_xyz.round(4)} "
                            f"rpy_deg={pred_rpy_deg.round(2)} "
                            f"grip_raw={pred_gripper_raw:.3f} grip01={g:.3f}"
                        )

                apply_gripper(model, data, grip_controls, lo_m, hi_m, g)

                q_target = dls_ik_compute_qtarget(
                    model, data, ee_body_id,
                    arm_joint_ids, qpos_adrs, dof_adrs,
                    pred_xyz, q_des,
                    rot_w=2.0,
                    damping=3e-2,
                    dt_step=dt_view,
                    qd_max=qd_max,
                    iters=IK_ITERS_TRANSFER,
                )
                data.ctrl[:7] = q_target

                for _ in range(n_sub_transfer):
                    mujoco.mj_step(model, data)

                viewer.sync()

                if cv2 is not None:
                    cv2.waitKey(1)

                if renderer is not None:
                    render_windows()

                step_realtime()
                step_count += 1

        except KeyboardInterrupt:
            pass
        finally:
            if cv2 is not None:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()