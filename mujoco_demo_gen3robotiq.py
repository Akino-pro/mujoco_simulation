"""
Gen3 + Robotiq-85 in MuJoCo: control by EEF pose sequence + 3-phase gripper (explicit mimic)
+ DLS IK with SAFE joint-speed limiting (10 deg/s)
+ Wrist camera window

ROBUST FIXES:
- INLINE robot <asset> children directly into the scene XML (no asset include).
- INLINE robot <worldbody> children directly under <body name="gen3_mount"> (no body include).
  (This avoids MuJoCo include XML single-root requirements and the "multiple root tags" issue.)

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


# ======== UPDATE THIS PATH ON YOUR MACHINE ========
URDF_PATH = r"D:\mujoco_simulation\gen3_modified.urdf"


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
    # We inline this inside <body name="gen3_mount"> ... </body>
    # So any valid body-children are fine; but make sure it's not comment-only/empty.
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
    urdf_abs = os.path.abspath(URDF_PATH)
    urdf_dir = os.path.dirname(urdf_abs)
    os.chdir(urdf_dir)

    # 1) Load URDF via MuJoCo importer
    robot_model = mujoco.MjModel.from_xml_path(urdf_abs)

    # 2) Export importer result as MJCF
    robot_mjcf_path = os.path.join(urdf_dir, "_gen3_from_urdf.xml")
    mujoco.mj_saveLastXML(robot_mjcf_path, robot_model)

    robot_text = Path(robot_mjcf_path).read_text(encoding="utf-8", errors="ignore")

    asset_inner = _extract_tag_inner(robot_text, "asset")
    worldbody_inner = _extract_tag_inner(robot_text, "worldbody")

    asset_children = _ensure_nonempty_asset_children(asset_inner)
    body_children  = _ensure_nonempty_body_children(worldbody_inner)

    # 3) Build the scene MJCF (assets + bodies INLINED)
    #scene_path = os.path.join(urdf_dir, "_gen3_scene_with_cubes.xml")
    scene_path = os.path.join(urdf_dir, "robotsuit_cubes.xml")

    #Path(scene_path).write_text(scene_xml, encoding="utf-8")
    #print("[DEBUG] scene_path =", scene_path)

    # 4) Load the scene
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgb")
    if wrist_cam_id < 0:
        raise RuntimeError("Camera 'wrist_rgb' not found in model (did you add it to the XML?)")
    wrist_cam_id = int(wrist_cam_id)

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

    cam_off = np.array([0.0, -0.06841, -0.05044], dtype=float)
    cam_rpy = np.array([0.0, 1.7444444444444447, -1.5708], dtype=float)

    def R_from_rpy(rpy: np.ndarray) -> np.ndarray:
        r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rx = np.array([[1, 0, 0],
                       [0, cr, -sr],
                       [0, sr,  cr]], dtype=float)
        Ry = np.array([[ cp, 0, sp],
                       [  0, 1,  0],
                       [-sp, 0, cp]], dtype=float)
        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]], dtype=float)
        return Rz @ Ry @ Rx

    R_cam_in_bracelet = R_from_rpy(cam_rpy)


    renderer = None
    if cv2 is None:
        print("[WARN] opencv-python not installed; wrist camera window disabled.")
    else:
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
    # -------------------------
    p_green = np.array([0.45, 0.18, 0.825], dtype=float)
    p_blue = np.array([0.45, -0.18, 0.825], dtype=float)

    z_above = 0.98
    z_pick = 0.86

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

    mid_z = 1.10
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
                            #draw_body_frame_in_viewer(viewer, data, ee_body_id, length=0.12, radius=0.003)
                            viewer.sync()

                            if renderer is not None:
                                # ====== SAME POSITION AS BEFORE ======
                                p_b = data.xpos[bracelet_body_id].copy()
                                R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
                                p_frame = p_b + R_b @ cam_off

                                R_frame = data.xmat[ee_body_id].reshape(3,
                                                                        3).copy()  # in your case ee is bracelet anyway

                                #eye_out = 0.035
                                eye_out = 0.01
                                eye_up = 0.000

                                tool_forward = np.array([0.0, 0.0, -1.0], dtype=float)
                                tool_up = np.array([0.0, -1.0, 0.0], dtype=float)

                                p_frame = p_frame + (R_frame @ (tool_forward * eye_out)) + (
                                            R_frame @ (tool_up * eye_up))

                                # ====== ORIENTATION THAT PRESERVES ROLL (NO "UP STABILIZATION") ======
                                def _normalize(v):
                                    n = np.linalg.norm(v)
                                    if n < 1e-9:
                                        return v
                                    return v / n

                                f_world = _normalize(R_frame @ tool_forward)  # where the tool points in world
                                u_world = _normalize(R_frame @ tool_up)  # tool-up in world (this carries the roll)

                                # MuJoCo camera looks along -Z, so z_cam_world should be -forward
                                z_cam_world = _normalize(-f_world)

                                # make y_cam_world as close to u_world as possible but orthogonal to z
                                y_cam_world = _normalize(u_world - np.dot(u_world, z_cam_world) * z_cam_world)

                                # x = y × z  (right-handed)
                                x_cam_world = _normalize(np.cross(y_cam_world, z_cam_world))

                                # Re-orthogonalize y to be safe
                                y_cam_world = _normalize(np.cross(z_cam_world, x_cam_world))

                                R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])  # world_from_cam

                                # Write camera pose (this keeps roll!)
                                model.cam_pos[wrist_cam_id] = p_frame
                                model.cam_quat[wrist_cam_id] = quat_from_mat3(R_wc)  # must output [w,x,y,z]

                                renderer.update_scene(data, camera="wrist_rgb")
                                wrist_rgb = renderer.render()
                                show_wrist_window(wrist_rgb, title="Wrist Camera (roll preserved)", w=480, h=480)

                                """
                                # ---- Wrist camera POSITION from URDF offset (bracelet + cam_off) ----
                                p_b = data.xpos[bracelet_body_id].copy()
                                R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
                                p_frame = p_b + R_b @ cam_off

                                # ---- Orientation: follow gripper/tool direction ----
                                # Use EE orientation so view aligns with gripper pointing direction
                                R_frame = data.xmat[ee_body_id].reshape(3, 3).copy()

                                # move camera origin outward so it’s not inside the wrist camera mesh
                                eye_out = 0.035  # 3.5 cm outward from mount point (tune 0.02~0.06)
                                eye_up = 0.008  # 8 mm upward (optional)

                                # tool-forward axis in the chosen frame (here: EE frame)
                                tool_forward = np.array([0.0, 0.0, -1.0])  # or whatever you’re using
                                tool_up = np.array([0.0, -1.0, 0.0])

                                p_frame = p_frame + (R_frame @ (tool_forward * eye_out)) + (
                                            R_frame @ (tool_up * eye_up))

                                cam = make_free_camera_from_frame_pose(
                                    p_frame, R_frame,
                                    # Gripper/tool axis: your print said index 0 was "up" initially -> likely X is tool axis.
                                    optical_forward=tool_forward,  # try [-1,0,0] if backwards
                                    optical_up= tool_up,  # try [0,-1,0] if rotated
                                    look_ahead=0.20,
                                    cam_back=0.02,
                                    keep_claws_down_bias=0.00,
                                    distance=0.20,
                                )
                                renderer.update_scene(data, camera=cam)
                                wrist_rgb = renderer.render()
                                show_wrist_window(wrist_rgb, title="Wrist Camera (Aligned)", w=480, h=360)
                                """

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
                            #draw_body_frame_in_viewer(viewer, data, ee_body_id, length=0.12, radius=0.003)
                            viewer.sync()

                            if renderer is not None:
                                # ====== SAME POSITION AS BEFORE ======
                                p_b = data.xpos[bracelet_body_id].copy()
                                R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
                                p_frame = p_b + R_b @ cam_off

                                R_frame = data.xmat[ee_body_id].reshape(3,
                                                                        3).copy()  # in your case ee is bracelet anyway

                                eye_out = 0.01
                                eye_up = 0.0

                                tool_forward = np.array([0.0, 0.0, -1.0], dtype=float)
                                tool_up = np.array([0.0, -1.0, 0.0], dtype=float)

                                p_frame = p_frame + (R_frame @ (tool_forward * eye_out)) + (
                                            R_frame @ (tool_up * eye_up))

                                # ====== ORIENTATION THAT PRESERVES ROLL (NO "UP STABILIZATION") ======
                                def _normalize(v):
                                    n = np.linalg.norm(v)
                                    if n < 1e-9:
                                        return v
                                    return v / n

                                f_world = _normalize(R_frame @ tool_forward)  # where the tool points in world
                                u_world = _normalize(R_frame @ tool_up)  # tool-up in world (this carries the roll)

                                # MuJoCo camera looks along -Z, so z_cam_world should be -forward
                                z_cam_world = _normalize(-f_world)

                                # make y_cam_world as close to u_world as possible but orthogonal to z
                                y_cam_world = _normalize(u_world - np.dot(u_world, z_cam_world) * z_cam_world)

                                # x = y × z  (right-handed)
                                x_cam_world = _normalize(np.cross(y_cam_world, z_cam_world))

                                # Re-orthogonalize y to be safe
                                y_cam_world = _normalize(np.cross(z_cam_world, x_cam_world))

                                R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])  # world_from_cam

                                # Write camera pose (this keeps roll!)
                                model.cam_pos[wrist_cam_id] = p_frame
                                model.cam_quat[wrist_cam_id] = quat_from_mat3(R_wc)  # must output [w,x,y,z]

                                renderer.update_scene(data, camera="wrist_rgb")
                                wrist_rgb = renderer.render()
                                show_wrist_window(wrist_rgb, title="Wrist Camera (roll preserved)", w=480, h=480)
                                """
                                # ---- Wrist camera POSITION from URDF offset (bracelet + cam_off) ----
                                p_b = data.xpos[bracelet_body_id].copy()
                                R_b = data.xmat[bracelet_body_id].reshape(3, 3).copy()
                                p_frame = p_b + R_b @ cam_off

                                # ---- Orientation: follow gripper/tool direction ----
                                # Use EE orientation so view aligns with gripper pointing direction
                                R_frame = data.xmat[ee_body_id].reshape(3, 3).copy()

                                # move camera origin outward so it’s not inside the wrist camera mesh
                                eye_out = 0.035  # 3.5 cm outward from mount point (tune 0.02~0.06)
                                eye_up = 0  # 8 mm upward (optional)

                                # tool-forward axis in the chosen frame (here: EE frame)
                                tool_forward = np.array([0.0, 0, -1.0])  # or whatever you’re using
                                tool_up = np.array([0.0, -1.0, 0.0])

                                p_frame = p_frame + (R_frame @ (tool_forward * eye_out)) + (
                                            R_frame @ (tool_up * eye_up))

                                cam = make_free_camera_from_frame_pose(
                                    p_frame, R_frame,
                                    # Gripper/tool axis: your print said index 0 was "up" initially -> likely X is tool axis.
                                    optical_forward=tool_forward,  # try [-1,0,0] if backwards
                                    optical_up=tool_up,  # try [0,-1,0] if rotated
                                    look_ahead=0.20,
                                    cam_back=0.02,
                                    keep_claws_down_bias=0.00,
                                    distance=0.20,
                                )
                                renderer.update_scene(data, camera=cam)
                                wrist_rgb = renderer.render()
                                show_wrist_window(wrist_rgb, title="Wrist Camera (Aligned)", w=480, h=360)
                                """

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
