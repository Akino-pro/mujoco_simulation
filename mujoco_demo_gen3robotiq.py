"""
Gen3 + Robotiq-85 in MuJoCo: control by EEF pose sequence + 3-phase gripper (explicit mimic)
+ DLS IK with SAFE joint-speed limiting (10 deg/s)
+ Wrist camera window
+ Fixed overhead camera window (looking straight down)

Notes:
- The MuJoCo viewer window remains interactive (you can move around).
- The wrist camera view is rendered via MuJoCo Renderer + OpenCV window (as before).
- The new fixed camera view is rendered from an MJCF camera named "fixed_down"
  (must exist in robotsuit_cubes.xml).

Install:
  pip install mujoco numpy opencv-python
"""
import os
import time
import re
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from helper_functions import (
    _get_joint_ids_by_name, find_ee_target, build_gripper_controls, quat_from_mat3,
    set_white_environment_visuals, ik_step_dls, apply_gripper, show_wrist_window,
    make_R_with_axis_k_down, make_free_camera_from_frame_pose, draw_body_frame_in_viewer
)

try:
    import cv2
except Exception:
    cv2 = None


# ======== UPDATE THIS PATH ON YOUR MACHINE (ONLY USED FOR WORKDIR) ========
URDF_PATH = r"gen3_modified.urdf"


def _extract_tag_inner(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    m2 = re.search(rf"<{tag}\b[^>]*/>", text, flags=re.DOTALL)
    if m2:
        return ""
    return ""


def _strip_comments(s: str) -> str:
    return re.sub(r"<!--.*?-->", "", s or "", flags=re.DOTALL)


def _ensure_nonempty_asset_children(asset_inner: str) -> str:
    keepalive = '<material name="__asset_keepalive" rgba="1 1 1 1"/>\n'
    s = (asset_inner or "").strip()
    if not s:
        return keepalive
    if not _strip_comments(s).strip():
        return keepalive
    return s + "\n"


def _ensure_nonempty_body_children(worldbody_inner: str) -> str:
    keepalive = '<geom name="__body_keepalive" type="sphere" size="0.0001" rgba="0 0 0 0"/>\n'
    s = (worldbody_inner or "").strip()
    if not s:
        return keepalive
    if not _strip_comments(s).strip():
        return keepalive
    return s + "\n"


def _indent(s: str, n_spaces: int) -> str:
    pad = " " * n_spaces
    lines = (s or "").splitlines()
    return "\n".join(pad + ln if ln.strip() else ln for ln in lines)


def main():
    # Keep your original behavior: cd into URDF directory so STL relative paths resolve.
    urdf_abs = os.path.abspath(URDF_PATH)
    urdf_dir = os.path.dirname(urdf_abs)
    os.chdir(urdf_dir)

    # Scene MJCF you showed (must contain cameras: "wrist_rgb" and "fixed_down")
    scene_path = os.path.join(urdf_dir, "robotsuit_cubes.xml")

    # 1) Load the scene
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 2) Camera IDs
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgb")
    if wrist_cam_id < 0:
        raise RuntimeError("Camera 'wrist_rgb' not found in model (did you add it to the XML?)")
    wrist_cam_id = int(wrist_cam_id)

    fixed_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_down")
    if fixed_cam_id < 0:
        raise RuntimeError("Camera 'fixed_down' not found in model (add it in <worldbody> of robotsuit_cubes.xml).")
    fixed_cam_id = int(fixed_cam_id)

    # -------------------------
    # Wrist camera body (if present)
    # -------------------------
    CAM_NAMES = [
        "wrist_mounted_camera_color_optical_frame",
        "wrist_mounted_camera_depth_optical_frame",
        "wrist_camera_link",
    ]
    cam_body_id = -1
    cam_body_name = None
    for nm in CAM_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            cam_body_id = int(bid)
            cam_body_name = nm
            break
    if cam_body_id >= 0:
        print(f"[INFO] Using wrist camera frame body: {cam_body_name}")
    else:
        print("[WARN] Wrist camera frame body not found. Falling back to bracelet+offset.")

    # -------------------------
    # Arm joints + EEF body
    # -------------------------
    arm_joint_names = [f"gen3_joint_{i}" for i in range(1, 8)]
    arm_joint_ids = _get_joint_ids_by_name(model, arm_joint_names)
    qpos_adrs = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
    dof_adrs = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]

    ee_body_id, ee_name = find_ee_target(model)
    print(f"[INFO] Using end-effector body '{ee_name}' (id={ee_body_id})")

    grip_controls, lo_m, hi_m = build_gripper_controls(model)

    bracelet_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gen3_bracelet_link")
    if bracelet_body_id < 0:
        raise RuntimeError("Could not find body 'gen3_bracelet_link' (needed for wrist camera fallback).")
    bracelet_body_id = int(bracelet_body_id)

    # Wrist camera extrinsics (fallback)
    cam_off = np.array([0.0, -0.06841, -0.05044], dtype=float)

    # Renderer
    renderer = None
    if cv2 is None:
        print("[WARN] opencv-python not installed; camera windows disabled.")
    else:
        # You can choose a larger resolution if you like
        renderer = mujoco.Renderer(model, width=480, height=480)

    # -------------------------
    # IK timing
    # -------------------------
    dt = 1.0 / 60.0
    IK_ITERS = 10
    qd_safe = np.deg2rad(10.0)
    qd_max = np.full(7, qd_safe, dtype=float)
    dt_inner = dt / IK_ITERS

    # -------------------------
    # Pick & place curve (green -> blue)
    # (NOTE: you updated table height to 0.72 in XML; cubes now at z=0.745)
    # -------------------------
    p_green = np.array([0.45, 0.18, 0.745], dtype=float)
    p_blue = np.array([0.45, -0.18, 0.745], dtype=float)

    z_above = 0.95
    z_pick = 0.80

    R0 = data.xmat[ee_body_id].reshape(3, 3).copy()
    R_des, k_up = make_R_with_axis_k_down(R0)
    q_hold = quat_from_mat3(R_des)
    print(f"[INFO] EE axis pointing up at start was body axis index {k_up} (0=x,1=y,2=z). Forcing it to point down.")

    def bezier(p0, p1, p2, p3, s):
        s = float(s)
        return ((1 - s) ** 3) * p0 + 3 * ((1 - s) ** 2) * s * p1 + 3 * (1 - s) * (s ** 2) * p2 + (s ** 3) * p3

    pA0 = np.array([p_green[0], p_green[1], z_above], dtype=float)
    pA1 = np.array([p_green[0], p_green[1], z_pick], dtype=float)
    pA2 = np.array([p_green[0], p_green[1], z_above], dtype=float)

    pB0 = np.array([p_blue[0], p_blue[1], z_above], dtype=float)
    pB1 = np.array([p_blue[0], p_blue[1], z_pick], dtype=float)
    pB2 = np.array([p_blue[0], p_blue[1], z_above], dtype=float)

    mid_z = 1.05
    c1 = np.array([p_green[0], 0.00, mid_z], dtype=float)
    c2 = np.array([p_blue[0], 0.00, mid_z], dtype=float)

    segments = [
        ("lin", pA0, pA0, 1.0, 0),
        ("lin", pA0, pA1, 1.2, 0),
        ("lin", pA1, pA1, 0.8, -1),
        ("lin", pA1, pA2, 1.2, 0),
        ("bez", pA2, c1, c2, pB0, 2.4, 0),
        ("lin", pB0, pB1, 1.2, 0),
        ("lin", pB1, pB1, 0.8, +1),
        ("lin", pB1, pB2, 1.0, 0),
        ("bez", pB2, c2, c1, pA0, 2.4, 0),
    ]

    g = 0.8
    GRIP_SPEED = 0.9

    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-9:
            return v
        return v / n

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 1.8
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        set_white_environment_visuals(model, viewer)

        next_step_time = time.perf_counter()

        try:
            while viewer.is_running():
                for seg in segments:
                    kind = seg[0]

                    if kind == "lin":
                        _, p0, p1, T, cmd = seg
                        T = float(T)
                        n = max(2, int(np.ceil(T / dt)))

                        for i in range(n):
                            if not viewer.is_running():
                                break

                            s = i / (n - 1)
                            p_des = (1.0 - s) * p0 + s * p1
                            q_des = q_hold

                            g += float(cmd) * GRIP_SPEED * dt
                            g = float(np.clip(g, 0.0, 1.0))
                            apply_gripper(model, data, grip_controls, lo_m, hi_m, g)

                            for _ in range(IK_ITERS):
                                err = ik_step_dls(
                                    model, data, ee_body_id,
                                    arm_joint_ids, qpos_adrs, dof_adrs,
                                    p_des, q_des,
                                    rot_w=2.0,
                                    damping=3e-2,
                                    dt_step=dt_inner,
                                    qd_max=qd_max,
                                )
                                if err < 1e-4:
                                    break

                            viewer.sync()

                            if renderer is not None:
                                # ===== Wrist camera (same behavior as before) =====
                                p_b = data.xpos[bracelet_body_id].copy()
                                R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
                                p_frame = p_b + R_b @ cam_off

                                R_frame = data.xmat[ee_body_id].reshape(3, 3).copy()

                                eye_out = 0.01
                                eye_up = 0.0
                                tool_forward = np.array([0.0, 0.0, -1.0], dtype=float)
                                tool_up = np.array([0.0, -1.0, 0.0], dtype=float)

                                p_frame = p_frame + (R_frame @ (tool_forward * eye_out)) + (R_frame @ (tool_up * eye_up))

                                f_world = _normalize(R_frame @ tool_forward)
                                u_world = _normalize(R_frame @ tool_up)

                                z_cam_world = _normalize(-f_world)
                                y_cam_world = _normalize(u_world - np.dot(u_world, z_cam_world) * z_cam_world)
                                x_cam_world = _normalize(np.cross(y_cam_world, z_cam_world))
                                y_cam_world = _normalize(np.cross(z_cam_world, x_cam_world))

                                R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])

                                model.cam_pos[wrist_cam_id] = p_frame
                                model.cam_quat[wrist_cam_id] = quat_from_mat3(R_wc)

                                renderer.update_scene(data, camera="wrist_rgb")
                                wrist_rgb = renderer.render()
                                wrist_bgr = wrist_rgb[..., ::-1]
                                show_wrist_window(wrist_bgr , title="Wrist Camera", w=480, h=480)

                                # ===== Fixed overhead camera window (MJCF camera fixed_down) =====
                                renderer.update_scene(data, camera="fixed_down")
                                fixed_rgb = renderer.render()
                                fixed_bgr = fixed_rgb[..., ::-1]
                                show_wrist_window(fixed_bgr, title="Fixed Camera (Downward)", w=480, h=480)

                            next_step_time += dt
                            sleep_s = next_step_time - time.perf_counter()
                            if sleep_s > 0:
                                time.sleep(sleep_s)

                    elif kind == "bez":
                        _, p0, p1, p2, p3, T, cmd = seg
                        T = float(T)
                        n = max(2, int(np.ceil(T / dt)))

                        for i in range(n):
                            if not viewer.is_running():
                                break

                            s = i / (n - 1)
                            p_des = bezier(p0, p1, p2, p3, s)
                            q_des = q_hold

                            g += float(cmd) * GRIP_SPEED * dt
                            g = float(np.clip(g, 0.0, 1.0))
                            apply_gripper(model, data, grip_controls, lo_m, hi_m, g)

                            for _ in range(IK_ITERS):
                                err = ik_step_dls(
                                    model, data, ee_body_id,
                                    arm_joint_ids, qpos_adrs, dof_adrs,
                                    p_des, q_des,
                                    rot_w=2.0,
                                    damping=3e-2,
                                    dt_step=dt_inner,
                                    qd_max=qd_max,
                                )
                                if err < 1e-4:
                                    break

                            viewer.sync()

                            if renderer is not None:
                                # ===== Wrist camera (same behavior as before) =====
                                p_b = data.xpos[bracelet_body_id].copy()
                                R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
                                p_frame = p_b + R_b @ cam_off

                                R_frame = data.xmat[ee_body_id].reshape(3, 3).copy()

                                eye_out = 0.01
                                eye_up = 0.0
                                tool_forward = np.array([0.0, 0.0, -1.0], dtype=float)
                                tool_up = np.array([0.0, -1.0, 0.0], dtype=float)

                                p_frame = p_frame + (R_frame @ (tool_forward * eye_out)) + (R_frame @ (tool_up * eye_up))

                                f_world = _normalize(R_frame @ tool_forward)
                                u_world = _normalize(R_frame @ tool_up)

                                z_cam_world = _normalize(-f_world)
                                y_cam_world = _normalize(u_world - np.dot(u_world, z_cam_world) * z_cam_world)
                                x_cam_world = _normalize(np.cross(y_cam_world, z_cam_world))
                                y_cam_world = _normalize(np.cross(z_cam_world, x_cam_world))

                                R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])

                                model.cam_pos[wrist_cam_id] = p_frame
                                model.cam_quat[wrist_cam_id] = quat_from_mat3(R_wc)

                                renderer.update_scene(data, camera="wrist_rgb")
                                wrist_rgb = renderer.render()
                                wrist_bgr = wrist_rgb[..., ::-1]
                                show_wrist_window(wrist_bgr , title="Wrist Camera", w=480, h=480)

                                # ===== Fixed overhead camera window =====
                                renderer.update_scene(data, camera="fixed_down")
                                fixed_rgb = renderer.render()
                                fixed_bgr = fixed_rgb[..., ::-1]
                                show_wrist_window(fixed_bgr, title="Fixed Camera (Downward)", w=480, h=480)

                            next_step_time += dt
                            sleep_s = next_step_time - time.perf_counter()
                            if sleep_s > 0:
                                time.sleep(sleep_s)

        except KeyboardInterrupt:
            pass
        finally:
            if cv2 is not None:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()